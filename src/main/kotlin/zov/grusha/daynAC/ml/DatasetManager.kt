package zov.grusha.daynAC.ml

import java.io.File
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.security.MessageDigest
import java.security.SecureRandom
import java.util.UUID

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

    private companion object {
        /** Колонки метаданных в конце строки векторного CSV (после признаков). */
        val META_COLUMNS = listOf("session_id", "player_hash")

        /** Версия признаков у датасетов, созданных до появления dataset.featver. */
        const val LEGACY_FEATURE_VERSION = 1
    }

    private val legitVectorFile = File(dataFolder, "legit_vectors.csv")
    private val cheatVectorFile = File(dataFolder, "cheat_vectors.csv")
    private val saltFile = File(dataFolder, "dataset.salt")

    /** Версия определения признаков, на которой записан датасет (см. [FeatureExtractor.FEATURE_VERSION]). */
    private val featureVersionFile = File(dataFolder, "dataset.featver")

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

    /** Результат парсинга отдельной строки CSV. */
    private sealed class ParseResult {
        object Ignored : ParseResult() // Заголовок или пустая строка
        class Success(val vector: DoubleArray, val sessionId: String?, val playerHash: String?) : ParseResult()
        class Error(val reason: String) : ParseResult()
    }

    /**
     * Единая функция парсинга строки векторного датасета. Принимает оба формата:
     * только признаки (старый) или признаки + session_id + player_hash (новый).
     */
    private fun parseVectorLine(line: String): ParseResult {
        val trimmed = line.trim()
        if (trimmed.isEmpty() || trimmed.startsWith("h0")) return ParseResult.Ignored

        val parts = trimmed.split(',')
        val hasMeta = parts.size == featureCount + META_COLUMNS.size
        if (parts.size != featureCount && !hasMeta) {
            return ParseResult.Error(
                "ожидалось $featureCount (или ${featureCount + META_COLUMNS.size} с метаданными) элементов, получено ${parts.size}"
            )
        }

        val vector = DoubleArray(featureCount)
        for (i in 0 until featureCount) {
            val v = parts[i].trim().toDoubleOrNull()
                ?: return ParseResult.Error("нечисловое значение '${parts[i]}' на позиции $i")
            vector[i] = v
        }

        val sessionId = if (hasMeta) parts[featureCount].trim().ifEmpty { null } else null
        val playerHash = if (hasMeta) parts[featureCount + 1].trim().ifEmpty { null } else null
        return ParseResult.Success(vector, sessionId, playerHash)
    }

    /**
     * Пишет вектор признаков окна ударов (windowSize * 21 + 7 значений) и две колонки
     * метаданных: [sessionId] (сессия записи) и [playerHash] (см. [playerHash]).
     * Метаданные в обучение не идут — по ним делится выборка на train/val/test.
     *
     * Вызывается только когда [FeatureExtractor.extract] вернул непустой вектор —
     * строк в векторных CSV меньше, чем в поштучных, это нормально.
     */
    fun writeVectorSample(label: String, vector: FeatureExtractor.FeatureVector, sessionId: String, playerHash: String) {
        val targetFile = when (label) {
            "legit" -> legitVectorFile
            "cheat" -> cheatVectorFile
            else -> return // неизвестная метка — не пишем никуда
        }

        // Запятые и переводы строк в идентификаторах сломали бы CSV
        fun clean(id: String) = id.replace(Regex("[,\\r\\n]"), "_")
        targetFile.appendText(vector.toCsvRow() + "," + clean(sessionId) + "," + clean(playerHash) + "\n")
    }

    /**
     * Читает все векторные сэмплы (legit_vectors.csv + cheat_vectors.csv) для обучения,
     * вместе с метаданными сессии и игрока.
     */
    fun readVectorSamples(): VectorDataset {
        val samples = mutableListOf<VectorSample>()
        var legitParsed = 0
        var cheatParsed = 0

        fun loadFile(file: File, label: Double) {
            if (!file.exists()) return
            var lineNumber = 0
            var skippedCount = 0
            val maxDetailedWarnings = 5

            fun logWarning(reason: String) {
                if (skippedCount < maxDetailedWarnings) {
                    System.err.println("[WARN] [${file.name}:$lineNumber] Пропуск: $reason")
                } else if (skippedCount == maxDetailedWarnings) {
                    System.err.println("[WARN] [${file.name}] Достигнут лимит подробных предупреждений ($maxDetailedWarnings). Дальнейшие пропуски выводятся только в итоговый счетчик.")
                }
            }

            file.forEachLine { line ->
                lineNumber++
                when (val result = parseVectorLine(line)) {
                    is ParseResult.Ignored -> return@forEachLine
                    is ParseResult.Error -> {
                        logWarning(result.reason)
                        skippedCount++
                    }
                    is ParseResult.Success -> {
                        samples.add(VectorSample(result.vector, label, result.sessionId, result.playerHash))
                        if (label == 0.0) legitParsed++ else cheatParsed++
                    }
                }
            }

            if (skippedCount > 0) {
                System.err.println("[INFO] [${file.name}] Всего пропущено некорректных строк: $skippedCount")
            }
        }

        loadFile(legitVectorFile, 0.0)
        loadFile(cheatVectorFile, 1.0)

        return VectorDataset(samples, legitParsed, cheatParsed)
    }

    /** Краткая статистика датасета для команды info (учитывает только валидные сэмплы). */
    fun getVectorDatasetStats(): Pair<Int, Int> {
        fun countValidSamples(file: File): Int =
            if (!file.exists()) 0
            else file.useLines { lines ->
                lines.count { line -> parseVectorLine(line) is ParseResult.Success }
            }

        return Pair(countValidSamples(legitVectorFile), countValidSamples(cheatVectorFile))
    }

    /**
     * Сколько разных сессий записи в каждом классе (legit, cheat). Строки старого
     * формата без session_id не считаются — у них нет группы.
     */
    fun getSessionStats(): Pair<Int, Int> {
        fun countSessions(file: File): Int =
            if (!file.exists()) 0
            else file.useLines { lines ->
                lines.mapNotNull { line -> (parseVectorLine(line) as? ParseResult.Success)?.sessionId }
                    .toSet()
                    .size
            }

        return Pair(countSessions(legitVectorFile), countSessions(cheatVectorFile))
    }
}