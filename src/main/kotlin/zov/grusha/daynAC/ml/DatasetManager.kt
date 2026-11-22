package zov.grusha.daynAC.ml

import java.io.File
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.security.MessageDigest
import java.security.SecureRandom
import java.util.UUID
import java.util.concurrent.ConcurrentLinkedQueue

class DatasetManager(
    private val dataFolder: File,
    private val windowSize: Int = 16
) {

    /**
     * Один вектор датасета вместе с метаданными группы.
     *
     * [sessionId] и [playerHash] нужны только для честного разбиения на train/val/test
     * (окна одного игрока или одной сессии очень похожи друг на друга) и НЕ входят
     * в признаки модели. null — строка из старого формата без метаданных.
     *
     * Класс оставлен для совместимости (тесты, ручный разбор CSV). Путь обучения
     * использует [TrainingDataset] — плоский массив без отдельного объекта на сэмпл.
     */
    class VectorSample(
        val features: DoubleArray,
        val label: Double,
        val sessionId: String?,
        val playerHash: String?
    )

    /** Весь векторный датасет: сэмплы обоих классов и их количество. */
    class VectorDataset(
        val samples: List<VectorSample>,
        val legitCount: Int,
        val cheatCount: Int
    )

    /** Сводка для /daynac info: считается за один проход по обоим CSV. */
    class DatasetSummary(
        val legitVectors: Int,
        val cheatVectors: Int,
        val legitSessions: Int,
        val cheatSessions: Int
    )

    /** Метаданные одной строки CSV: группы для разбиения. */
    private data class Meta(val sessionId: String?, val playerHash: String?)

    private companion object {
        /** Колонки метаданных в конце строки векторного CSV (после признаков). */
        val META_COLUMNS = listOf("session_id", "player_hash")

        /** Версия признаков у датасетов, созданных до появления dataset.featver. */
        const val LEGACY_FEATURE_VERSION = 1

        /** Сколько подробных предупреждений о битых строках выводить за файл. */
        const val MAX_DETAILED_WARNINGS = 5
    }

    private val legitVectorFile = File(dataFolder, "legit_vectors.csv")
    private val cheatVectorFile = File(dataFolder, "cheat_vectors.csv")
    private val saltFile = File(dataFolder, "dataset.salt")

    /** Версия определения признаков, на которой записан датасет (см. [FeatureExtractor.FEATURE_VERSION]). */
    private val featureVersionFile = File(dataFolder, "dataset.featver")

    /** Строки, ожидающие записи. Publication через ConcurrentLinkedQueue даёт happens-before. */
    private data class PendingWrite(
        val file: File,
        val vector: FeatureExtractor.FeatureVector,
        val sessionId: String,
        val playerHash: String
    )

    private val pendingWrites = ConcurrentLinkedQueue<PendingWrite>()

    /** Число числовых признаков в одной строке вектора (без метаданных). */
    private val featureCount = windowSize * FeatureExtractor.PER_HIT_FEATURES + FeatureExtractor.AGGREGATE_FEATURES

    /** Соль для player_hash: хэши стабильны на этом сервере, но не раскрывают UUID. */
    private val salt: String

    init {
        if (!dataFolder.exists()) dataFolder.mkdirs()
        archiveDatasetIfFeatureVersionChanged()
        if (!legitVectorFile.exists()) legitVectorFile.writeText(vectorHeader() + "\n") else migrateLegacyVectorFile(legitVectorFile)
        if (!cheatVectorFile.exists()) cheatVectorFile.writeText(vectorHeader() + "\n") else migrateLegacyVectorFile(cheatVectorFile)
        salt = loadOrCreateSalt()
    }

    /**
     * Датасет хранит уже ВЫЧИСЛЕННЫЕ признаки, а не сырые дельты поворота, поэтому при смене
     * определения признаков (см. [FeatureExtractor.FEATURE_VERSION]) старые строки пересчитать
     * нельзя, а смешивать их с новыми — значит учить модель на несопоставимых значениях.
     *
     * Если версия в dataset.featver отличается от текущей (нет файла = версия 1, датасет
     * создан до появления версионирования), файлы с данными переименовываются в
     * `<имя>.featv<N>.bak` и датасет начинается с чистого листа. Ничего не удаляется.
     */
    private fun archiveDatasetIfFeatureVersionChanged() {
        val current = FeatureExtractor.FEATURE_VERSION
        val stored = featureVersionFile.takeIf { it.exists() }
            ?.readText()?.trim()?.toIntOrNull()
            ?: LEGACY_FEATURE_VERSION

        if (stored != current) {
            for (file in listOf(legitVectorFile, cheatVectorFile)) {
                if (!file.exists() || !hasDataRows(file)) continue
                var backup = File(file.parentFile, "${file.name}.featv$stored.bak")
                if (backup.exists()) backup = File(file.parentFile, "${file.name}.featv$stored.${System.currentTimeMillis()}.bak")
                Files.move(file.toPath(), backup.toPath())
                System.err.println(
                    "[INFO] [${file.name}] версия признаков изменилась ($stored → $current): " +
                            "старые данные сохранены как ${backup.name}, датасет начат заново."
                )
            }
        }
        if (stored != current || !featureVersionFile.exists()) {
            featureVersionFile.writeText(current.toString())
        }
    }

    /** Есть ли в CSV хотя бы одна строка кроме заголовка. */
    private fun hasDataRows(file: File): Boolean =
        file.useLines { lines -> lines.drop(1).any { it.isNotBlank() } }

    private fun vectorHeader(): String =
        FeatureExtractor.csvHeader(windowSize) + "," + META_COLUMNS.joinToString(",")

    private fun loadOrCreateSalt(): String {
        if (saltFile.exists()) {
            val existing = saltFile.readText().trim()
            if (existing.isNotEmpty()) return existing
        }
        val bytes = ByteArray(16)
        SecureRandom().nextBytes(bytes)
        val created = bytes.joinToString("") { "%02x".format(it) }
        saltFile.writeText(created)
        return created
    }

    /**
     * Псевдоним игрока для датасета: первые 12 hex-символов SHA-256(соль + UUID).
     * Один и тот же игрок всегда получает один и тот же хэш на этом сервере (пока цела
     * dataset.salt), поэтому по нему можно группировать данные, но CSV можно показывать
     * другим, не раскрывая UUID.
     */
    fun playerHash(playerId: UUID): String {
        val digest = MessageDigest.getInstance("SHA-256")
            .digest((salt + playerId.toString()).toByteArray(Charsets.UTF_8))
        return digest.take(6).joinToString("") { "%02x".format(it) }
    }

    /**
     * Приводит векторный CSV старого формата (только признаки) к новому: добавляет колонки
     * session_id и player_hash. Старые строки получают пустые метаданные — у них нет группы,
     * и при разбиении они считаются независимыми окнами, как раньше. Перед миграцией
     * один раз создаётся копия файла (*.bak).
     */
    private fun migrateLegacyVectorFile(file: File) {
        val firstLine = file.useLines { it.firstOrNull() }
        if (firstLine == null) {
            file.writeText(vectorHeader() + "\n")
            return
        }
        if (firstLine.contains("session_id")) return      // уже новый формат
        if (!firstLine.trim().startsWith("h0")) return    // неизвестный формат — не трогаем

        val backup = File(file.parentFile, file.name + ".bak")
        if (!backup.exists()) file.copyTo(backup)

        val tmp = File(file.parentFile, file.name + ".tmp")
        var migrated = 0
        tmp.bufferedWriter().use { out ->
            out.write(vectorHeader() + "\n")
            var isFirstLine = true
            file.forEachLine { raw ->
                if (isFirstLine) { isFirstLine = false; return@forEachLine }
                val line = raw.trim()
                if (line.isEmpty()) return@forEachLine
                if (line.split(',').size == featureCount) {
                    out.write("$line,,\n")
                    migrated++
                } else {
                    out.write(line + "\n") // нестандартную строку не трогаем — парсер её пропустит с предупреждением
                }
            }
        }
        Files.move(tmp.toPath(), file.toPath(), StandardCopyOption.REPLACE_EXISTING)
        System.err.println(
            "[INFO] [${file.name}] формат обновлён: добавлены колонки ${META_COLUMNS.joinToString(", ")} " +
                    "(строк без группы: $migrated). Копия старого файла: ${backup.name}"
        )
    }

    /** Есть ли в строке ровно featureCount (или featureCount + 2) колонок. */
    private fun columnCount(line: String): Int = line.trim().split(',').size

    private fun hasMetaColumns(columns: Int): Boolean = columns == featureCount + META_COLUMNS.size

    /** Быстрая проверка формата строки без парсинга чисел (для подсчёта и сводки). */
    private fun isValidColumns(columns: Int): Boolean =
        columns == featureCount || hasMetaColumns(columns)

    /**
     * Парсит строку прямо в плоский массив: ни одного `DoubleArray` на сэмпл.
     * @return метаданные строки; некорректная строка выбрасывает [IllegalArgumentException]
     */
    private fun parseInto(line: String, out: DoubleArray, offset: Int): Meta {
        val trimmed = line.trim()
        val parts = trimmed.split(',')
        val hasMeta = hasMetaColumns(parts.size)
        if (!isValidColumns(parts.size)) {
            throw IllegalArgumentException(
                "ожидалось $featureCount (или ${featureCount + META_COLUMNS.size} с метаданными) элементов, получено ${parts.size}"
            )
        }
        for (i in 0 until featureCount) {
            val v = parts[i].trim().toDoubleOrNull()
                ?: throw IllegalArgumentException("нечисловое значение '${parts[i]}' на позиции $i")
            out[offset + i] = v
        }
        return Meta(
            sessionId = if (hasMeta) parts[featureCount].trim().ifEmpty { null } else null,
            playerHash = if (hasMeta) parts[featureCount + 1].trim().ifEmpty { null } else null
        )
    }

    /**
     * Читает оба CSV в ОДИН плоский [TrainingDataset]. Делает два прохода по файлам:
     * первый — подсчёт валидных строк, чтобы выделить массивы точного размера, второй —
     * парсинг. Диск читается в фоновом потоке обучения; на главном потоке работы нет.
     */
    @Synchronized
    fun readCompactDataset(): TrainingDataset {
        val legitRows = countValidRows(legitVectorFile)
        val cheatRows = countValidRows(cheatVectorFile)
        val total = legitRows + cheatRows

        val features = DoubleArray(total * featureCount)
        val labels = DoubleArray(total)
        val sessions = arrayOfNulls<String>(total)
        val hashes = arrayOfNulls<String>(total)

        var written = 0
        fun loadFile(file: File, label: Double) {
            if (!file.exists()) return
            var lineNumber = 0
            var skipped = 0
            fun warn(reason: String) {
                if (skipped < MAX_DETAILED_WARNINGS) {
                    System.err.println("[WARN] [${file.name}:$lineNumber] Пропуск: $reason")
                } else if (skipped == MAX_DETAILED_WARNINGS) {
                    System.err.println(
                        "[WARN] [${file.name}] Достигнут лимит подробных предупреждений ($MAX_DETAILED_WARNINGS). " +
                                "Дальнейшие пропуски выводятся только в итоговый счетчик."
                    )
                }
            }
            file.forEachLine { line ->
                lineNumber++
                val trimmed = line.trim()
                if (trimmed.isEmpty() || trimmed.startsWith("h0")) return@forEachLine
                val meta = try {
                    parseInto(line, features, written * featureCount)
                } catch (e: IllegalArgumentException) {
                    warn(e.message ?: "некорректная строка")
                    skipped++
                    return@forEachLine
                }
                labels[written] = label
                sessions[written] = meta.sessionId
                hashes[written] = meta.playerHash
                written++
            }
            if (skipped > 0) {
                System.err.println("[INFO] [${file.name}] Всего пропущено некорректных строк: $skipped")
            }
        }

        loadFile(legitVectorFile, 0.0)
        loadFile(cheatVectorFile, 1.0)

        // Битые строки, найденные только при парсинге чисел, ужимаются здесь.
        val needCopy = written != total
        return TrainingDataset(
            features = if (needCopy) features.copyOf(written * featureCount) else features,
            labels = if (needCopy) labels.copyOf(written) else labels,
            sessionIds = if (needCopy) sessions.copyOf(written) else sessions,
            playerHashes = if (needCopy) hashes.copyOf(written) else hashes,
            sampleCount = written,
            legitCount = labels.count { it == 0.0 },
            cheatCount = labels.count { it == 1.0 }
        )
    }

    private fun countValidRows(file: File): Int =
        if (!file.exists()) 0
        else file.useLines { lines ->
            lines.count { line ->
                val trimmed = line.trim()
                trimmed.isNotEmpty() && !trimmed.startsWith("h0") && isValidColumns(columnCount(trimmed))
            }
        }

    /**
     * Читает все векторные сэмплы (legit_vectors.csv + cheat_vectors.csv) — совместимый
     * с прежним API вариант [readCompactDataset]. Само обучение этим путём не идёт:
     * здесь на каждый сэмпл создаётся отдельный DoubleArray, что дорого по памяти.
     */
    fun readVectorSamples(): VectorDataset {
        val compact = readCompactDataset()
        val samples = ArrayList<VectorSample>(compact.sampleCount)
        for (i in 0 until compact.sampleCount) {
            val from = i * featureCount
            samples.add(
                VectorSample(
                    compact.features.copyOfRange(from, from + featureCount),
                    compact.labels[i],
                    compact.sessionIds[i],
                    compact.playerHashes[i]
                )
            )
        }
        return VectorDataset(samples, compact.legitCount, compact.cheatCount)
    }

    /** Краткая статистика датасета для команды info (учитывает только валидные сэмплы). */
    fun getVectorDatasetStats(): Pair<Int, Int> {
        val s = getDatasetSummary()
        return s.legitVectors to s.cheatVectors
    }

    /**
     * Сколько разных сессий записи в каждом классе (legit, cheat). Строки старого
     * формата без session_id не считаются — у них нет группы.
     */
    fun getSessionStats(): Pair<Int, Int> {
        val s = getDatasetSummary()
        return s.legitSessions to s.cheatSessions
    }

    /**
     * Всё, что показывает /daynac info, за один проход по каждому файлу вместо
     * четырёх (прежде stats и sessions читали CSV независимо). Вызывается на
     * фоновом потоке: диск не должен блокировать тик сервера.
     */
    @Synchronized
    fun getDatasetSummary(): DatasetSummary {
        fun scan(file: File): Pair<Int, Int> {
            if (!file.exists()) return 0 to 0
            return file.useLines { lines ->
                var vectors = 0
                val sessions = HashSet<String>()
                for (line in lines) {
                    val trimmed = line.trim()
                    if (trimmed.isEmpty() || trimmed.startsWith("h0")) continue
                    val parts = trimmed.split(',')
                    if (!isValidColumns(parts.size)) continue
                    vectors++
                    if (hasMetaColumns(parts.size)) {
                        val session = parts[featureCount].trim()
                        if (session.isNotEmpty()) sessions.add(session)
                    }
                }
                vectors to sessions.size
            }
        }

        val (legitVectors, legitSessions) = scan(legitVectorFile)
        val (cheatVectors, cheatSessions) = scan(cheatVectorFile)
        return DatasetSummary(legitVectors, cheatVectors, legitSessions, cheatSessions)
    }

    /** Запятые и переводы строк в идентификаторах сломали бы CSV. */
    private fun cleanCsv(id: String): String = id.replace(Regex("[,\\r\\n]"), "_")

    private fun csvRow(vector: FeatureExtractor.FeatureVector, sessionId: String, playerHash: String): String =
        vector.toCsvRow() + "," + cleanCsv(sessionId) + "," + cleanCsv(playerHash) + "\n"

    private fun targetFileFor(label: String): File? = when (label) {
        "legit" -> legitVectorFile
        "cheat" -> cheatVectorFile
        else -> null
    }

    /**
     * Немедленная запись (для тестов и ручных вызовов). Основной путь боёвого сервера —
     * [enqueueVectorSample]: открывать файл на каждый 16-й удар в тике нельзя.
     */
    @Synchronized
    fun writeVectorSample(label: String, vector: FeatureExtractor.FeatureVector, sessionId: String, playerHash: String) {
        val targetFile = targetFileFor(label) ?: return
        targetFile.appendText(csvRow(vector, sessionId, playerHash))
    }

    /**
     * Кладывает вектор в буфер без диска и без форматирования строки — вызывается из
     * обработчика удара на главном потоке. Файл открывается один раз на пачку в
     * [flushPendingWrites], который вызывается фоновым планировщиком плагина.
     */
    fun enqueueVectorSample(label: String, vector: FeatureExtractor.FeatureVector, sessionId: String, playerHash: String) {
        val targetFile = targetFileFor(label) ?: return
        pendingWrites.add(PendingWrite(targetFile, vector, sessionId, playerHash))
    }

    /** Сколько векторов ждёт записи (для диагностики и тестов). */
    fun pendingWriteCount(): Int = pendingWrites.size

    /**
     * Сливает накопленные строки в CSV: по одному append на файл, а не на сэмпл.
     * @return число записанных векторов
     */
    @Synchronized
    fun flushPendingWrites(): Int {
        if (pendingWrites.isEmpty()) return 0
        val byFile = HashMap<File, StringBuilder>()
        var count = 0
        while (true) {
            val entry = pendingWrites.poll() ?: break
            byFile.getOrPut(entry.file) { StringBuilder() }
                .append(csvRow(entry.vector, entry.sessionId, entry.playerHash))
            count++
        }
        for ((file, text) in byFile) file.appendText(text.toString())
        return count
    }
}
