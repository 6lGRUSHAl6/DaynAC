package zov.grusha.daynAC.ml

import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.Closeable
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.EOFException
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.FilterInputStream
import java.io.IOException
import java.io.InputStream
import java.io.OutputStream
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.security.MessageDigest
import java.util.UUID
import java.util.zip.CRC32
import java.util.zip.Deflater
import java.util.zip.DeflaterOutputStream
import java.util.zip.DataFormatException
import java.util.zip.Inflater

/**
 * Свой архиватор векторных CSV. Главное требование — не « compact ради compact», а полная
 * обратимость: после [import] файлы датасета побайтово равны исходным, а значит обучение
 * увидит ровно те же числа, метки, метаданные и порядок строк.
 *
 * Почему не просто gzip:
 *  - в архиве лежит манифест (версия признаков, размер окна, число строк, SHA-256 каждого
 *    исходного CSV) — битый или несовместимый архив не сможет молча заменить рабочий датасет;
 *  - числовые строки кодируются битами double (What You See Is What You Train), строки,
 *    которые не восстанавливаются побайтово, падают в raw-чанк без потери точности;
 *  - каждый чанк имеет CRC32, весь архив — SHA-256, а импорт проверяет восстановленные
 *    файлы по хэшам из манифеста ещё до подмены рабочих файлов;
 *  - подмена идёт через tmp + atomic move с бэкапами и откатом при любой ошибке.
 *
 * Архив НЕ является рабочим хранилищем: плагин продолжает дописывать CSV. Экспорт берёт
 * тот же мьютекс, что и запись/чтение датасета, и перед снимком сбрасывает буфер записи.
 */
class DatasetArchive(private val manager: DatasetManager) {

    data class ExportOptions(
        /** Соль нужна, чтобы player_hash из архива совпадал с хэшами новых игроков. */
        val includeSalt: Boolean = true
    )

    data class ImportOptions(
        /** Разрешить замену локальной dataset.salt (меняет хэши ВСЕХ будущих игроков). */
        val replaceSalt: Boolean = false
    )

    data class ExportResult(
        val file: File,
        val bytes: Long,
        val numericRows: Int,
        val rawLines: Int,
        val includedSalt: Boolean
    )

    data class VerifyResult(
        val featureVersion: Int,
        val windowSize: Int,
        val featureCount: Int,
        val legitRows: Int,
        val cheatRows: Int,
        val numericRows: Int,
        val rawLines: Int,
        val includedSalt: Boolean
    )

    data class ImportResult(
        val numericRows: Int,
        val rawLines: Int,
        val saltReplaced: Boolean,
        val backups: List<File>
    )

    private enum class Kind(val id: Int, val fileName: String) {
        LEGIT(0, "legit_vectors.csv"),
        CHEAT(1, "cheat_vectors.csv");

        companion object {
            fun of(id: Int): Kind = entries.firstOrNull { it.id == id }
                ?: throw IOException("неизвестный тип файла в архиве: $id")
        }
    }

    private data class Header(
        val flags: Int,
        val featureVersion: Int,
        val windowSize: Int,
        val featureCount: Int,
        val legitRows: Int,
        val cheatRows: Int,
        val createdAtMs: Long,
        val saltDigest: ByteArray
    )

    private class FileMeta(
        val kind: Kind,
        val length: Long,
        val sha: ByteArray,
        val planLength: Int,
        val numericRows: Int,
        val rawLines: Int
    )

    private class ParsedLine(
        val body: String,
        val eol: Int,
        val values: DoubleArray,
        val metaPresent: Boolean,
        val meta1: ByteArray,
        val meta2: ByteArray
    )

    private class NumericRow(
        val values: DoubleArray,
        val metaPresent: Boolean,
        val meta1: ByteArray,
        val meta2: ByteArray,
        val eol: Int
    )

    private data class Counts(val numeric: Int, val raw: Int)

    private class Decoded(
        val header: Header,
        val staged: Map<Kind, File>,
        val numericRows: Int,
        val rawLines: Int,
        val salt: String?
    )

    companion object {
        private val MAGIC = "DAYNAC01".toByteArray(Charsets.US_ASCII)
        private val END_MAGIC = "DACEND01".toByteArray(Charsets.US_ASCII)
        private const val FORMAT_VERSION = 1

        private const val CHUNK_END = 0
        private const val CHUNK_FILE = 1
        private const val CHUNK_PLAN = 2
        private const val CHUNK_NUMERIC = 3
        private const val CHUNK_RAW = 4
        private const val CHUNK_SALT = 5

        private const val FLAG_SALT = 1

        private const val NUMERIC_BLOCK_ROWS = 128
        private const val RAW_FLUSH_BYTES = 1 shl 20
        private const val MAX_PAYLOAD = 64 * 1024 * 1024
        private const val MAX_PLAN = 16 * 1024 * 1024
        private const val MAX_LINE = 16 * 1024 * 1024

        /** План строк файла: 0 = числовая строка, 1 = исходная строка. */
        private fun fileFor(kind: Kind, manager: DatasetManager): File = when (kind) {
            Kind.LEGIT -> manager.legitVectorFile
            Kind.CHEAT -> manager.cheatVectorFile
        }

        private fun eolText(code: Int): String = when (code) {
            0 -> "\n"
            1 -> "\r\n"
            2 -> ""
            else -> throw IOException("неизвестный признак перевода строки: $code")
        }

        private fun reconstructBody(row: NumericRow): String = buildString {
            append(row.values.joinToString(",") { it.toString() })
            if (row.metaPresent) {
                append(',').append(String(row.meta1, Charsets.UTF_8))
                append(',').append(String(row.meta2, Charsets.UTF_8))
            }
        }

        private fun reconstruct(row: NumericRow): String = reconstructBody(row) + eolText(row.eol)

        private fun sha256(bytes: ByteArray): ByteArray =
            MessageDigest.getInstance("SHA-256").digest(bytes)

        private fun deflate(data: ByteArray): ByteArray {
            // DeflaterOutputStream не закрывает переданный ему Deflater: без end() нативная память
            // zlib (сотни КБ на чанк) висит до сборки мусора, а на большом датасете чанков тысячи.
            val deflater = Deflater(Deflater.DEFAULT_COMPRESSION, true)
            try {
                val out = ByteArrayOutputStream(data.size / 2 + 64)
                DeflaterOutputStream(out, deflater).use { it.write(data) }
                return out.toByteArray()
            } finally {
                deflater.end()
            }
        }

        /** Метаданные строки не должны содержать разделителей: иначе число колонок «поплывёт». */
        private fun isSafeMeta(bytes: ByteArray): Boolean =
            bytes.none { it == ','.code.toByte() || it == '\n'.code.toByte() || it == '\r'.code.toByte() }

        private fun inflate(data: ByteArray, expected: Int): ByteArray {
            val inflater = Inflater(true)
            try {
                inflater.setInput(data)
                // Не выделяем заявленный размер заранее: битый архив не должен съесть память.
                val out = ByteArrayOutputStream(minOf(expected, 1 shl 20))
                val buffer = ByteArray(8192)
                while (!inflater.finished()) {
                    val n = try {
                        inflater.inflate(buffer)
                    } catch (e: DataFormatException) {
                        throw IOException("повреждённый сжатый чанк: ${e.message}", e)
                    }
                    if (n > 0) {
                        if (out.size() + n > expected) throw IOException("чанк больше объявленного размера")
                        out.write(buffer, 0, n)
                    } else if (inflater.needsInput() || inflater.needsDictionary()) {
                        throw IOException("оборванный сжатый чанк")
                    }
                }
                if (out.size() != expected) throw IOException(
                    "размер чанка ${out.size()} не совпадает с объявленным $expected"
                )
                if (inflater.remaining != 0) throw IOException("лишние байты после сжатого чанка")
                return out.toByteArray()
            } finally {
                inflater.end()
            }
        }

        /** Разбивает поток на строки, сохраняя их исходные байты вместе с переводом строки. */
        private inline fun forEachLine(input: InputStream, action: (ByteArray) -> Unit) {
            val line = ByteArrayOutputStream(8192)
            val block = ByteArray(64 * 1024)
            while (true) {
                val n = input.read(block)
                if (n < 0) break
                var from = 0
                for (i in 0 until n) {
                    if (block[i] == '\n'.code.toByte()) {
                        line.write(block, from, i - from + 1)
                        if (line.size() > MAX_LINE) throw IOException("слишком длинная строка CSV")
                        action(line.toByteArray())
                        line.reset()
                        from = i + 1
                    }
                }
                line.write(block, from, n - from)
                if (line.size() > MAX_LINE) throw IOException("слишком длинная строка CSV")
            }
            if (line.size() > 0) action(line.toByteArray())
        }

        /**
         * Разбирает строку так же, как readCompactDataset(): заголовок, пустые строки и
         * строки с нечисловыми/неверным числом колонок training не видит — они уходят в raw.
         */
        private fun parseLine(raw: ByteArray, featureCount: Int): ParsedLine? {
            val text = String(raw, Charsets.UTF_8)
            val trimmed = text.trim()
            if (trimmed.isEmpty() || trimmed.startsWith("h0")) return null
            val parts = trimmed.split(',')
            val hasMeta = parts.size == featureCount + 2
            if (parts.size != featureCount && !hasMeta) return null
            val values = DoubleArray(featureCount)
            for (i in 0 until featureCount) {
                values[i] = parts[i].trim().toDoubleOrNull() ?: return null
            }
            val eol = when {
                text.endsWith("\r\n") -> 1
                text.endsWith("\n") -> 0
                else -> 2
            }
            val body = text.dropLast(eolText(eol).length)
            return ParsedLine(
                body = body,
                eol = eol,
                values = values,
                metaPresent = hasMeta,
                meta1 = if (hasMeta) parts[featureCount].trim().toByteArray(Charsets.UTF_8) else ByteArray(0),
                meta2 = if (hasMeta) parts[featureCount + 1].trim().toByteArray(Charsets.UTF_8) else ByteArray(0)
            )
        }

        /** Числовая строка идёт в бинарный чанк, только если её печать даёт те же байты. */
        private fun toNumericRow(parsed: ParsedLine): NumericRow? {
            val row = NumericRow(
                values = parsed.values,
                metaPresent = parsed.metaPresent,
                meta1 = parsed.meta1,
                meta2 = parsed.meta2,
                eol = parsed.eol
            )
            return row.takeIf {
                reconstructBody(it).toByteArray(Charsets.UTF_8).contentEquals(parsed.body.toByteArray(Charsets.UTF_8))
            }
        }
    }

    // ---------------------------------------------------------------- экспорт

    private class Snapshot(val header: Header, val numeric: Int, val raw: Int)

    fun export(archive: File, options: ExportOptions = ExportOptions()): ExportResult =
        manager.runExclusiveArchiveOp {
            cleanupStaleTemp()
            val parent = archive.absoluteFile.parentFile
            parent.mkdirs()
            val tmp = File(parent, "${archive.name}.${UUID.randomUUID()}.tmp")
            val checkDir = Files.createTempDirectory(manager.dataFolder.toPath(), ".dac-check-").toFile()
            try {
                // Под мьютексом датасета — только снимок: сброс буфера записи и чтение CSV.
                val snapshot = manager.withDatasetLock {
                    manager.flushPendingWrites()
                    writeSnapshot(tmp, options)
                }

                // Публикуем только архив, который уже удалось прочитать и проверить. Проверка читает
                // лишь временный файл, поэтому запись датасета и /daynac info не ждут её окончания.
                val decoded = decode(tmp, checkDir, materialize = false)
                check(decoded.numericRows == snapshot.numeric && decoded.rawLines == snapshot.raw) {
                    "внутренняя ошибка: число строк в архиве не совпало с записанным"
                }
                check(
                    decoded.header.legitRows == snapshot.header.legitRows &&
                            decoded.header.cheatRows == snapshot.header.cheatRows
                ) {
                    "внутренняя ошибка: число строк обучения изменилось во время экспорта"
                }
                atomicMove(tmp, archive)
                ExportResult(archive, archive.length(), snapshot.numeric, snapshot.raw, decoded.salt != null)
            } finally {
                tmp.delete()
                checkDir.deleteRecursively()
            }
        }

    /** Пишет архив во временный файл. Вызывается под мьютексом датасета. */
    private fun writeSnapshot(tmp: File, options: ExportOptions): Snapshot {
        // Читаем соль один раз: флаг и хэш в манифесте должны описывать один и тот же чанк.
        val salt = if (options.includeSalt) readSalt() else null
        val header = Header(
            flags = if (salt != null) FLAG_SALT else 0,
            featureVersion = currentFeatureVersion(),
            windowSize = manager.windowSize,
            featureCount = manager.featureCount,
            legitRows = countTrainingRows(fileFor(Kind.LEGIT, manager), manager.featureCount),
            cheatRows = countTrainingRows(fileFor(Kind.CHEAT, manager), manager.featureCount),
            createdAtMs = System.currentTimeMillis(),
            saltDigest = salt?.let { sha256(it.toByteArray(Charsets.UTF_8)) } ?: ByteArray(32)
        )

        var numeric = 0
        var raw = 0
        FileOutputStream(tmp).use { fos ->
            BufferedOutputStream(fos).use { rawOut ->
                val digest = MessageDigest.getInstance("SHA-256")
                val out = DataOutputStream(HashingOutputStream(rawOut, digest))
                writeHeader(out, header)
                val writer = ChunkWriter(out)
                for (kind in Kind.entries) {
                    val counts = writeKind(writer, kind, fileFor(kind, manager), manager.featureCount)
                    numeric += counts.numeric
                    raw += counts.raw
                }
                salt?.let { writer.write(CHUNK_SALT, it.toByteArray(Charsets.UTF_8)) }
                out.writeByte(CHUNK_END)
                out.flush()
                rawOut.write(digest.digest())
                rawOut.write(END_MAGIC)
                rawOut.flush()
                // Архив переживает потерю питания: проверенный файл не должен оказаться обрезанным.
                fos.fd.sync()
            }
        }
        return Snapshot(header, numeric, raw)
    }

    /** Проверяет архив, не трогая рабочие CSV и не записывая на диск восстановленные копии. */
    fun verify(archive: File): VerifyResult {
        cleanupStaleTemp()
        val checkDir = Files.createTempDirectory(manager.dataFolder.toPath(), ".dac-verify-").toFile()
        try {
            val decoded = decode(archive, checkDir, materialize = false)
            return VerifyResult(
                featureVersion = decoded.header.featureVersion,
                windowSize = decoded.header.windowSize,
                featureCount = decoded.header.featureCount,
                legitRows = decoded.header.legitRows,
                cheatRows = decoded.header.cheatRows,
                numericRows = decoded.numericRows,
                rawLines = decoded.rawLines,
                includedSalt = decoded.salt != null
            )
        } finally {
            checkDir.deleteRecursively()
        }
    }

    /** Остатки прерванных операций (процесс убили): удаляем только старые, чужие не трогаем. */
    private fun cleanupStaleTemp() {
        val threshold = System.currentTimeMillis() - 60 * 60 * 1000L
        manager.dataFolder.listFiles()?.forEach { f ->
            if (f.lastModified() >= threshold) return@forEach
            if (f.isDirectory && f.name.startsWith(".dac-")) f.deleteRecursively()
            else if (f.isFile && f.name.contains(".dac.") && f.name.endsWith(".tmp")) f.delete()
        }
    }

    private fun writeHeader(out: DataOutputStream, header: Header) {
        out.write(MAGIC)
        out.writeInt(FORMAT_VERSION)
        out.writeInt(header.flags)
        out.writeInt(header.featureVersion)
        out.writeInt(header.windowSize)
        out.writeInt(header.featureCount)
        out.writeInt(header.legitRows)
        out.writeInt(header.cheatRows)
        out.writeLong(header.createdAtMs)
        out.write(header.saltDigest)
    }

    private fun writeKind(writer: ChunkWriter, kind: Kind, file: File, featureCount: Int): Counts {
        val fileDigest = MessageDigest.getInstance("SHA-256")
        val plan = ByteArrayOutputStream()
        var numeric = 0
        var raw = 0
        val numericBlock = ArrayList<NumericRow>(NUMERIC_BLOCK_ROWS)
        val rawChunk = ByteArrayOutputStream()
        val rawData = DataOutputStream(rawChunk)
        var rawInChunk = 0

        fun flushNumeric() {
            if (numericBlock.isEmpty()) return
            val out = ByteArrayOutputStream()
            val data = DataOutputStream(out)
            data.writeByte(kind.id)
            data.writeInt(numericBlock.size)
            for (row in numericBlock) {
                data.writeByte(if (row.metaPresent) 1 else 0)
                if (row.metaPresent) {
                    data.writeInt(row.meta1.size); data.write(row.meta1)
                    data.writeInt(row.meta2.size); data.write(row.meta2)
                }
                data.writeByte(row.eol)
            }
            // По столбцам: соседние значения одного признака лучше сжимаются вместе.
            for (f in 0 until featureCount) {
                for (row in numericBlock) data.writeLong(row.values[f].toBits())
            }
            data.flush()
            writer.write(CHUNK_NUMERIC, out.toByteArray())
            numeric += numericBlock.size
            numericBlock.clear()
        }

        fun flushRaw() {
            if (rawInChunk == 0) return
            val out = ByteArrayOutputStream()
            val data = DataOutputStream(out)
            data.writeByte(kind.id)
            data.writeInt(rawInChunk)
            data.write(rawChunk.toByteArray())
            data.flush()
            writer.write(CHUNK_RAW, out.toByteArray())
            raw += rawInChunk
            rawChunk.reset()
            rawInChunk = 0
        }

        fun addRaw(line: ByteArray) {
            rawData.writeInt(line.size)
            rawData.write(line)
            rawInChunk++
            plan.write(1)
            if (rawChunk.size() >= RAW_FLUSH_BYTES) flushRaw()
        }

        // Длину считаем по реально прочитанным байтам (а не file.length() после чтения), а отсутствующий
        // файл трактуем как пустой — раньше его удаление «на лету» роняло экспорт FileNotFoundException.
        var length = 0L
        if (file.exists()) FileInputStream(file).use { input ->
            forEachLine(input) { line ->
                fileDigest.update(line)
                length += line.size
                val numericRow = parseLine(line, featureCount)?.let(::toNumericRow)
                if (numericRow != null) {
                    numericBlock.add(numericRow)
                    plan.write(0)
                    if (numericBlock.size >= NUMERIC_BLOCK_ROWS) flushNumeric()
                } else {
                    addRaw(line)
                }
            }
        }
        flushNumeric()
        flushRaw()

        val meta = ByteArrayOutputStream()
        val data = DataOutputStream(meta)
        data.writeByte(kind.id)
        data.writeLong(length)
        data.write(fileDigest.digest())
        data.writeInt(plan.size())
        data.writeInt(numeric)
        data.writeInt(raw)
        writer.write(CHUNK_FILE, meta.toByteArray())

        val planPayload = ByteArray(plan.size() + 1)
        planPayload[0] = kind.id.toByte()
        System.arraycopy(plan.toByteArray(), 0, planPayload, 1, plan.size())
        writer.write(CHUNK_PLAN, planPayload)
        return Counts(numeric, raw)
    }

    // ---------------------------------------------------------------- импорт

    fun import(archive: File, options: ImportOptions = ImportOptions()): ImportResult =
        manager.runExclusiveArchiveOp { restore(archive, options) }

    private fun restore(archive: File, options: ImportOptions): ImportResult {
        cleanupStaleTemp()
        val workDir = Files.createTempDirectory(manager.dataFolder.toPath(), ".dac-import-").toFile()
        try {
            val decoded = decode(archive, workDir)
            val version = currentFeatureVersion()
            require(decoded.header.featureVersion == version) {
                "архив собран с версией признаков ${decoded.header.featureVersion}, " +
                        "текущая версия $version — восстановление отменено"
            }
            require(decoded.header.windowSize == manager.windowSize) {
                "архив собран с окном ${decoded.header.windowSize} ударов, текущее окно ${manager.windowSize}"
            }
            require(decoded.header.featureCount == manager.featureCount) {
                "архив содержит ${decoded.header.featureCount} признаков, текущая схема ${manager.featureCount}"
            }
            val incomingSalt = decoded.salt?.trim()?.takeIf { it.isNotEmpty() }

            return manager.withDatasetLock {
                // Накопленные векторы дописываем в ТЕКУЩИЕ файлы: они уйдут в бэкап, а не смешаются
                // с восстановленным датасетом.
                manager.flushPendingWrites()

                // Соль читаем уже под мьютексом: между проверкой и заменой она не должна измениться.
                // Сравнение по обрезанным строкам, как в DatasetManager.loadOrCreateSalt().
                val localSalt = readSalt()
                val replaceSalt = incomingSalt != null && incomingSalt != localSalt
                require(!replaceSalt || localSalt == null || options.replaceSalt) {
                    "соль в архиве отличается от dataset.salt на сервере; " +
                            "замена соли изменит player_hash всех новых игроков — нужна явная пометка --replace-salt"
                }

                val backups = ArrayList<File>()
                val backupOf = HashMap<File, File>()
                val restored = LinkedHashSet<File>()
                fun makeBackup(file: File) {
                    if (!file.exists()) return
                    val backup = Files.createTempFile(
                        manager.dataFolder.toPath(), file.name + ".preimport-", ".bak"
                    ).toFile()
                    Files.copy(file.toPath(), backup.toPath(), StandardCopyOption.REPLACE_EXISTING)
                    backups.add(backup)
                    backupOf[file] = backup
                }

                try {
                    for (kind in Kind.entries) makeBackup(fileFor(kind, manager))
                    if (replaceSalt) makeBackup(manager.saltFile)

                    for (kind in Kind.entries) {
                        val target = fileFor(kind, manager)
                        atomicMove(decoded.staged[kind] ?: error("нет восстановленного файла $kind"), target)
                        restored.add(target)
                    }
                    if (replaceSalt && incomingSalt != null) {
                        val tmpSalt = File(workDir, "dataset.salt.restored")
                        tmpSalt.writeText(incomingSalt)
                        atomicMove(tmpSalt, manager.saltFile)
                        restored.add(manager.saltFile)
                        manager.reloadSalt()
                    }
                    ImportResult(decoded.numericRows, decoded.rawLines, replaceSalt, backups.toList())
                } catch (e: Exception) {
                    // Любая ошибка после первой подмены возвращает прежние файлы. Ошибка самого отката
                    // не должна затирать исходную причину — она уходит в suppressed.
                    var rollbackFailed = false
                    for (target in restored) {
                        try {
                            val backup = backupOf[target]
                            if (backup != null && backup.exists()) atomicMove(backup, target)
                            else target.delete() // до импорта файла не было
                        } catch (rollbackError: Exception) {
                            rollbackFailed = true
                            e.addSuppressed(rollbackError)
                        }
                    }
                    if (manager.saltFile in restored) runCatching { manager.reloadSalt() }
                    // Если откат не удался, бэкап — единственная копия: его не трогаем.
                    if (!rollbackFailed) backups.forEach { it.delete() }
                    throw e
                }
            }
        } finally {
            workDir.deleteRecursively()
        }
    }

    // ---------------------------------------------------------------- чтение архива

    private fun decode(archive: File, workDir: File, materialize: Boolean = true): Decoded {
        require(archive.exists()) { "архив не найден: ${archive.name}" }
        val digest = MessageDigest.getInstance("SHA-256")
        val metas = HashMap<Kind, FileMeta>()
        val plans = HashMap<Kind, ByteArray>()
        val numericSpools = HashMap<Kind, LineSpool>()
        val rawSpools = HashMap<Kind, LineSpool>()
        val numericCounts = HashMap<Kind, Int>()
        val rawCounts = HashMap<Kind, Int>()
        var salt: String? = null
        var header: Header? = null

        try {
            try {
                FileInputStream(archive).buffered(64 * 1024).use { rawIn ->
                    val input = DataInputStream(HashingInputStream(rawIn, digest))
                    val magic = ByteArray(MAGIC.size)
                    input.readFully(magic)
                    require(magic.contentEquals(MAGIC)) { "это не архив DaynAC (битая сигнатура)" }
                    val formatVersion = input.readInt()
                    require(formatVersion == FORMAT_VERSION) { "неподдерживаемая версия формата архива: $formatVersion" }
                    val flags = input.readInt()
                    require(flags and FLAG_SALT.inv() == 0) { "неизвестные флаги архива: $flags" }
                    val featureVersion = input.readInt()
                    val windowSize = input.readInt()
                    val featureCount = input.readInt()
                    val legitRows = input.readInt()
                    val cheatRows = input.readInt()
                    val createdAt = input.readLong()
                    val saltDigest = ByteArray(32)
                    input.readFully(saltDigest)
                    require(featureVersion > 0 && windowSize > 0 && featureCount > 0) { "некорректный заголовок архива" }
                    require(legitRows >= 0 && cheatRows >= 0) { "некорректное число строк в заголовке" }
                    require(featureCount == windowSize * FeatureExtractor.PER_HIT_FEATURES + FeatureExtractor.AGGREGATE_FEATURES) {
                        "число признаков $featureCount не соответствует окну $windowSize"
                    }
                    header = Header(flags, featureVersion, windowSize, featureCount, legitRows, cheatRows, createdAt, saltDigest)

                    while (true) {
                        val type = input.readUnsignedByte()
                        if (type == CHUNK_END) break
                        require(type in CHUNK_FILE..CHUNK_SALT) { "неизвестный тип чанка: $type" }
                        val payloadLength = input.readInt()
                        val storedLength = input.readInt()
                        val expectedCrc = input.readInt()
                        require(payloadLength in 0..MAX_PAYLOAD) { "некорректный размер чанка: $payloadLength" }
                        require(storedLength in 0..(MAX_PAYLOAD * 2)) { "некорректный сжатый размер чанка: $storedLength" }
                        val compressed = ByteArray(storedLength)
                        input.readFully(compressed)
                        val payload = inflate(compressed, payloadLength)
                        val crc = CRC32().apply { update(payload) }.value.toInt()
                        require(crc == expectedCrc) { "CRC чанка $type не совпал — архив повреждён" }

                        when (type) {
                            CHUNK_FILE -> {
                                val data = DataInputStream(ByteArrayInputStream(payload))
                                val kind = Kind.of(data.readUnsignedByte())
                                val length = data.readLong()
                                val sha = ByteArray(32)
                                data.readFully(sha)
                                val planLength = data.readInt()
                                val numericRows = data.readInt()
                                val rawLines = data.readInt()
                                require(metas[kind] == null) { "дубликат метаданных файла $kind" }
                                require(length >= 0 && planLength in 0..MAX_PLAN && numericRows >= 0 && rawLines >= 0) {
                                    "некорректные метаданные файла $kind"
                                }
                                metas[kind] = FileMeta(kind, length, sha, planLength, numericRows, rawLines)
                            }
                            CHUNK_PLAN -> {
                                require(payload.isNotEmpty()) { "пустой план строк" }
                                val kind = Kind.of(payload[0].toInt())
                                require(plans[kind] == null) { "дубликат плана строк $kind" }
                                val plan = payload.copyOfRange(1, payload.size)
                                require(plan.size <= MAX_PLAN) { "слишком большой план строк" }
                                plans[kind] = plan
                            }
                            CHUNK_NUMERIC -> {
                                val data = DataInputStream(ByteArrayInputStream(payload))
                                val kind = Kind.of(data.readUnsignedByte())
                                val rowCount = data.readInt()
                                val fc = header!!.featureCount
                                require(rowCount in 1..MAX_PAYLOAD / (fc * 8 + 16)) { "некорректное число строк в блоке: $rowCount" }
                                val metaPresent = BooleanArray(rowCount)
                                val meta1 = Array(rowCount) { ByteArray(0) }
                                val meta2 = Array(rowCount) { ByteArray(0) }
                                val eols = ByteArray(rowCount)
                                for (r in 0 until rowCount) {
                                    val metaFlag = data.readUnsignedByte()
                                    require(metaFlag == 0 || metaFlag == 1) { "некорректный флаг метаданных строки: $metaFlag" }
                                    metaPresent[r] = metaFlag == 1
                                    if (metaPresent[r]) {
                                        meta1[r] = readLengthPrefixed(data, MAX_LINE)
                                        meta2[r] = readLengthPrefixed(data, MAX_LINE)
                                        require(isSafeMeta(meta1[r]) && isSafeMeta(meta2[r])) {
                                            "недопустимые символы в метаданных строки"
                                        }
                                    }
                                    eols[r] = data.readByte()
                                }
                                val values = Array(rowCount) { DoubleArray(fc) }
                                for (f in 0 until fc) {
                                    for (r in 0 until rowCount) values[r][f] = Double.fromBits(data.readLong())
                                }
                                val spool = numericSpools.getOrPut(kind) { LineSpool(workDir, "numeric-${kind.id}.bin") }
                                repeat(rowCount) { r ->
                                    val row = NumericRow(values[r], metaPresent[r], meta1[r], meta2[r], eols[r].toInt())
                                    spool.add(reconstruct(row).toByteArray(Charsets.UTF_8))
                                }
                                numericCounts[kind] = (numericCounts[kind] ?: 0) + rowCount
                            }
                            CHUNK_RAW -> {
                                val data = DataInputStream(ByteArrayInputStream(payload))
                                val kind = Kind.of(data.readUnsignedByte())
                                val lineCount = data.readInt()
                                require(lineCount in 1..MAX_PAYLOAD / 4) { "некорректное число сырых строк: $lineCount" }
                                val spool = rawSpools.getOrPut(kind) { LineSpool(workDir, "raw-${kind.id}.bin") }
                                repeat(lineCount) {
                                    spool.add(readLengthPrefixed(data, MAX_LINE))
                                }
                                rawCounts[kind] = (rawCounts[kind] ?: 0) + lineCount
                            }
                            CHUNK_SALT -> {
                                require(salt == null) { "дубликат соли в архиве" }
                                require(flags and FLAG_SALT != 0) { "соль есть, но флаг соли не выставлен" }
                                val value = String(payload, Charsets.UTF_8)
                                require(sha256(payload).contentEquals(header!!.saltDigest)) { "SHA-256 соли не совпал с манифестом" }
                                salt = value
                            }
                        }
                    }

                    val storedDigest = ByteArray(32)
                    // Хэш считался по всем байтам до конца чанков, включая CHUNK_END.
                    // Концевые байты читаем мимо хэширующего потока — они в хэш не входят.
                    val footer = DataInputStream(rawIn)
                    footer.readFully(storedDigest)
                    val endMagic = ByteArray(END_MAGIC.size)
                    footer.readFully(endMagic)
                    require(storedDigest.contentEquals(digest.digest())) { "SHA-256 архива не совпал — архив повреждён" }
                    require(endMagic.contentEquals(END_MAGIC)) { "отсутствует концевая метка архива" }
                    require(rawIn.read() == -1) { "после архива есть лишние байты" }
                }
            } catch (e: EOFException) {
                throw IOException("архив оборван или обрезан", e)
            }

            val h = header ?: throw IOException("в архиве нет заголовка")
            require((h.flags and FLAG_SALT) == 0 || salt != null) { "в манифесте есть соль, но в архиве её нет" }
            // Спулы писались через буфер — закрываем до чтения, иначе хвост строк потеряется.
            numericSpools.values.forEach { it.close() }
            rawSpools.values.forEach { it.close() }

            val staged = HashMap<Kind, File>()
            val trainingRows = HashMap<Kind, Int>()
            var totalNumeric = 0
            var totalRaw = 0
            for (kind in Kind.entries) {
                val meta = metas[kind] ?: throw IOException("в архиве нет файла ${kind.fileName}")
                val plan = plans[kind] ?: throw IOException("в архиве нет плана строк ${kind.fileName}")
                require(plan.size == meta.planLength) { "длина плана строк ${kind.fileName} не совпала с манифестом" }
                // Сверяем фактическое число строк в чанках, а не только наличие спула: лишние строки
                // в архиве молча игнорировались бы планом.
                require((numericCounts[kind] ?: 0) == meta.numericRows) { "число числовых строк ${kind.fileName} не совпало" }
                require((rawCounts[kind] ?: 0) == meta.rawLines) { "число сырых строк ${kind.fileName} не совпало" }
                var planNumeric = 0
                var planRaw = 0
                for (code in plan) {
                    when (code.toInt()) {
                        0 -> planNumeric++
                        1 -> planRaw++
                        else -> throw IOException("неизвестный код в плане ${kind.fileName}")
                    }
                }
                require(planNumeric == meta.numericRows) { "план ${kind.fileName}: numeric mismatch" }
                require(planRaw == meta.rawLines) { "план ${kind.fileName}: raw mismatch" }

                val numericSpool = numericSpools[kind]
                val rawSpool = rawSpools[kind]
                val fileDigest = MessageDigest.getInstance("SHA-256")
                var restoredLength = 0L
                // Числовые строки по построению читаются обучением; сырые проверяем тем же парсером,
                // что и readCompactDataset(): заголовки и пустые строки обучение пропускает. Так число
                // строк обучения считается за тот же проход, что и хэш, без повторного чтения файла.
                var training = meta.numericRows

                // Для verify и самопроверки экспорта восстановленные байты только хэшируются, не пишутся на диск.
                val target = if (materialize) File(workDir, kind.name + ".restored") else null
                val fos = target?.let { FileOutputStream(it) }
                val sink: OutputStream = if (fos != null) BufferedOutputStream(fos) else OutputStream.nullOutputStream()
                sink.use { out ->
                    val numericReader = numericSpool?.reader()
                    val rawReader = rawSpool?.reader()
                    try {
                        for (code in plan) {
                            val isRaw = code.toInt() == 1
                            val line = (if (isRaw) rawReader else numericReader)?.next()
                                ?: throw IOException("в плане есть строка, но чанк отсутствует")
                            fileDigest.update(line)
                            restoredLength += line.size
                            if (isRaw && parseLine(line, h.featureCount) != null) training++
                            out.write(line)
                        }
                    } finally {
                        numericReader?.close()
                        rawReader?.close()
                    }
                    out.flush()
                    fos?.fd?.sync()
                }
                require(restoredLength == meta.length) { "размер ${kind.fileName} не совпал с манифестом" }
                require(fileDigest.digest().contentEquals(meta.sha)) {
                    "SHA-256 восстановленного файла ${kind.fileName} не совпал с манифестом"
                }
                if (target != null) staged[kind] = target
                trainingRows[kind] = training
                totalNumeric += meta.numericRows
                totalRaw += meta.rawLines
            }
            require(trainingRows[Kind.LEGIT] == h.legitRows && trainingRows[Kind.CHEAT] == h.cheatRows) {
                "число строк обучения в архиве (${trainingRows[Kind.LEGIT]}/${trainingRows[Kind.CHEAT]}) " +
                        "не совпало с манифестом (${h.legitRows}/${h.cheatRows})"
            }
            return Decoded(h, staged, totalNumeric, totalRaw, salt)
        } finally {
            // Дескрипторы закрываем при любом исходе: на Windows открытый спул не даст удалить workDir.
            numericSpools.values.forEach { runCatching { it.close() } }
            rawSpools.values.forEach { runCatching { it.close() } }
        }
    }

    private fun readLengthPrefixed(data: DataInputStream, max: Int): ByteArray {
        val size = data.readInt()
        require(size in 0..max) { "некорректная длина строки в архиве: $size" }
        val bytes = ByteArray(size)
        data.readFully(bytes)
        return bytes
    }

    // ---------------------------------------------------------------- утилиты

    private fun currentFeatureVersion(): Int = manager.storedFeatureVersion()

    private fun countTrainingRows(file: File, featureCount: Int): Int {
        if (!file.exists()) return 0
        var count = 0
        FileInputStream(file).use { input ->
            forEachLine(input) { line ->
                if (parseLine(line, featureCount) != null) count++
            }
        }
        return count
    }

    /** Соль в том же виде, в каком её держит DatasetManager (loadOrCreateSalt обрезает пробелы). */
    private fun readSalt(): String? =
        manager.saltFile.takeIf { it.exists() }?.readText()?.trim()?.takeIf { it.isNotEmpty() }

    private fun atomicMove(source: File, target: File) {
        target.parentFile?.mkdirs()
        try {
            Files.move(
                source.toPath(), target.toPath(),
                StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING
            )
        } catch (e: AtomicMoveNotSupportedException) {
            Files.move(source.toPath(), target.toPath(), StandardCopyOption.REPLACE_EXISTING)
        }
    }

    private class ChunkWriter(private val out: DataOutputStream) {
        fun write(type: Int, payload: ByteArray) {
            val compressed = deflate(payload)
            out.writeByte(type)
            out.writeInt(payload.size)
            out.writeInt(compressed.size)
            out.writeInt(CRC32().apply { update(payload) }.value.toInt())
            out.write(compressed)
        }
    }

    private class HashingOutputStream(
        private val out: OutputStream,
        private val digest: MessageDigest
    ) : OutputStream() {
        override fun write(b: Int) {
            out.write(b)
            digest.update(b.toByte())
        }

        override fun write(b: ByteArray, off: Int, len: Int) {
            out.write(b, off, len)
            digest.update(b, off, len)
        }

        override fun flush() = out.flush()
        override fun close() = out.close()
    }

    private class HashingInputStream(
        input: InputStream,
        private val digest: MessageDigest
    ) : FilterInputStream(input) {
        override fun read(): Int {
            val value = super.read()
            if (value >= 0) digest.update(value.toByte())
            return value
        }

        override fun read(b: ByteArray, off: Int, len: Int): Int {
            val count = super.read(b, off, len)
            if (count > 0) digest.update(b, off, count)
            return count
        }
    }

    private class LineSpool(workDir: File, name: String) : Closeable {
        private val file = File(workDir, name)
        private val out = DataOutputStream(BufferedOutputStream(FileOutputStream(file)))

        fun add(line: ByteArray) {
            out.writeInt(line.size)
            out.write(line)
        }

        fun reader(): LineReader = LineReader(DataInputStream(BufferedInputStream(FileInputStream(file))))

        private var closed = false

        /** Идемпотентно: спул закрывают и штатно перед чтением, и в finally при ошибке. */
        override fun close() {
            if (closed) return
            closed = true
            out.close()
        }
    }

    private class LineReader(private val input: DataInputStream) : Closeable {
        fun next(): ByteArray {
            val size = input.readInt()
            require(size in 0..MAX_LINE) { "некорректная длина строки в архиве: $size" }
            val bytes = ByteArray(size)
            input.readFully(bytes)
            return bytes
        }

        override fun close() = input.close()
    }
}