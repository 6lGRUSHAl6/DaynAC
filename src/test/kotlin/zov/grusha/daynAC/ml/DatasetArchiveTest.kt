package zov.grusha.daynAC.ml

import org.junit.jupiter.api.io.TempDir
import java.io.File
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue
import kotlin.test.fail

class DatasetArchiveTest {

    private fun vector(seed: Int): FeatureExtractor.FeatureVector =
        FeatureExtractor.FeatureVector(DoubleArray(343) { (it * seed % 97) / 97.0 }, 16)

    private fun fill(dir: File, rows: Int = 40) {
        val dm = DatasetManager(dir, windowSize = 16)
        repeat(rows) { i ->
            dm.writeVectorSample("legit", vector(i + 1), "session-${i % 4}", "hash-${i % 3}")
            dm.writeVectorSample("cheat", vector(i + 50), "session-${i % 5}", "hash-${i % 2}")
        }
    }

    private fun csv(dir: File): List<File> = listOf(
        File(dir, "legit_vectors.csv"),
        File(dir, "cheat_vectors.csv")
    )

    /** ByteArray.equals сравнивает ссылки, поэтому сравниваем содержимое явно. */
    private fun assertSameBytes(expected: List<ByteArray?>, actual: List<ByteArray?>) {
        assertEquals(expected.size, actual.size)
        expected.zip(actual).forEachIndexed { index, (a, b) ->
            when {
                a == null && b == null -> Unit
                a != null && b != null && a.contentEquals(b) -> Unit
                else -> fail("файл $index изменился")
            }
        }
    }

    @Test
    fun `export import restores both csv byte for byte`(@TempDir source: Path, @TempDir target: Path) {
        val src = source.toFile()
        fill(src)
        // Нетипичные строки: пустая, битовая, с «не каноничным» числом и CRLF.
        File(src, "legit_vectors.csv").apply {
            appendText("\n")
            appendText("1,2,3\n")
            appendText("0.100000000000000005," + "0.0,".repeat(340) + "s,h\r\n")
        }

        val archive = File(src, "dataset.dac")
        val exported = DatasetArchive(DatasetManager(src, 16)).export(archive)
        assertTrue(exported.numericRows > 0)
        assertTrue(exported.rawLines >= 3)

        DatasetManager(target.toFile(), 16) // создаёт файлы-заглушки, которые архив заменит
        DatasetArchive(DatasetManager(target.toFile(), 16)).import(
            archive, DatasetArchive.ImportOptions(replaceSalt = true)
        )

        csv(src).zip(csv(target.toFile())).forEach { (a, b) ->
            assertTrue(a.readBytes().contentEquals(b.readBytes()), "${a.name} должен восстановиться побайтово")
        }

        val original = DatasetManager(src, 16).readCompactDataset()
        val restored = DatasetManager(target.toFile(), 16).readCompactDataset()
        assertEquals(original.sampleCount, restored.sampleCount)
        assertTrue(original.features.contentEquals(restored.features))
        assertTrue(original.labels.contentEquals(restored.labels))
        assertTrue(original.sessionIds.contentEquals(restored.sessionIds))
        assertTrue(original.playerHashes.contentEquals(restored.playerHashes))
    }

    @Test
    fun `export actually compresses a decent dataset`(@TempDir dir: Path) {
        val dataDir = dir.toFile()
        fill(dataDir, rows = 300)
        val archive = File(dataDir, "dataset.dac")
        val info = DatasetArchive(DatasetManager(dataDir, 16)).export(archive)
        val csvBytes = csv(dataDir).sumOf { it.length() }
        assertTrue(info.numericRows > 0)
        assertTrue(archive.length() < csvBytes / 2, "архив ${archive.length()} должен быть меньше $csvBytes / 2")
    }

    @Test
    fun `corrupted archive is rejected and leaves dataset untouched`(@TempDir source: Path, @TempDir target: Path) {
        fill(source.toFile(), rows = 10)
        val archive = File(source.toFile(), "dataset.dac")
        DatasetArchive(DatasetManager(source.toFile(), 16)).export(archive)

        val bytes = archive.readBytes()
        bytes[bytes.size / 2] = (bytes[bytes.size / 2].toInt() xor 0x40).toByte()
        archive.writeBytes(bytes)

        val targetDm = DatasetManager(target.toFile(), 16)
        val before = csv(target.toFile()).map { if (it.exists()) it.readBytes() else null }
        assertFailsWith<Exception> {
            DatasetArchive(targetDm).import(archive)
        }
        val after = csv(target.toFile()).map { if (it.exists()) it.readBytes() else null }
        assertSameBytes(before, after)
    }

    @Test
    fun `truncated archive is rejected`(@TempDir dir: Path) {
        fill(dir.toFile(), rows = 10)
        val archive = File(dir.toFile(), "dataset.dac")
        DatasetArchive(DatasetManager(dir.toFile(), 16)).export(archive)
        val bytes = archive.readBytes()
        archive.writeBytes(bytes.copyOf(bytes.size - 5))
        assertFailsWith<Exception> {
            DatasetArchive(DatasetManager(dir.toFile(), 16)).verify(archive)
        }
    }

    @Test
    fun `import refuses incompatible feature version`(@TempDir source: Path, @TempDir target: Path) {
        fill(source.toFile(), rows = 10)
        val archive = File(source.toFile(), "dataset.dac")
        DatasetArchive(DatasetManager(source.toFile(), 16)).export(archive)

        val targetDir = target.toFile()
        val targetDm = DatasetManager(targetDir, 16)
        val before = csv(targetDir).map { it.readBytes() }
        File(targetDir, "dataset.featver").writeText((FeatureExtractor.FEATURE_VERSION + 1).toString())

        assertFailsWith<IllegalArgumentException> {
            DatasetArchive(targetDm).import(archive)
        }
        assertSameBytes(before, csv(targetDir).map { it.readBytes() })
    }

    @Test
    fun `salt mismatch needs explicit replacement`(@TempDir source: Path, @TempDir target: Path) {
        fill(source.toFile(), rows = 10)
        val archive = File(source.toFile(), "dataset.dac")
        DatasetArchive(DatasetManager(source.toFile(), 16)).export(archive)

        val targetDir = target.toFile()
        val targetDm = DatasetManager(targetDir, 16)
        val uuid = java.util.UUID.randomUUID()
        val sourceHash = DatasetManager(source.toFile(), 16).playerHash(uuid)
        assertTrue(sourceHash != targetDm.playerHash(uuid))

        assertFailsWith<IllegalArgumentException> {
            DatasetArchive(targetDm).import(archive)
        }
        DatasetArchive(targetDm).import(archive, DatasetArchive.ImportOptions(replaceSalt = true))
        assertEquals(sourceHash, targetDm.playerHash(uuid))
        assertTrue(csv(source.toFile())[0].readBytes().contentEquals(csv(targetDir)[0].readBytes()))
    }

    @Test
    fun `export without salt keeps the target salt intact`(@TempDir source: Path, @TempDir target: Path) {
        fill(source.toFile(), rows = 5)
        val archive = File(source.toFile(), "nosalt.dac")
        val exported = DatasetArchive(DatasetManager(source.toFile(), 16)).export(
            archive, DatasetArchive.ExportOptions(includeSalt = false)
        )
        assertTrue(!exported.includedSalt)

        val targetDir = target.toFile()
        val targetDm = DatasetManager(targetDir, 16)
        val saltBefore = File(targetDir, "dataset.salt").readBytes()
        val imported = DatasetArchive(targetDm).import(archive)

        assertTrue(!imported.saltReplaced)
        assertTrue(saltBefore.contentEquals(File(targetDir, "dataset.salt").readBytes()))
        assertSameBytes(csv(source.toFile()).map { it.readBytes() }, csv(targetDir).map { it.readBytes() })
    }

    @Test
    fun `verify reports manifest without touching dataset`(@TempDir dir: Path) {
        fill(dir.toFile(), rows = 20)
        val archive = File(dir.toFile(), "dataset.dac")
        val dm = DatasetManager(dir.toFile(), 16)
        val before = csv(dir.toFile()).map { it.readBytes() }
        val info = DatasetArchive(dm).export(archive)
        val verified = DatasetArchive(dm).verify(archive)

        assertEquals(FeatureExtractor.FEATURE_VERSION, verified.featureVersion)
        assertEquals(343, verified.featureCount)
        assertEquals(20, verified.legitRows)
        assertEquals(20, verified.cheatRows)
        assertEquals(info.numericRows, verified.numericRows)
        assertTrue(verified.includedSalt)
        assertSameBytes(before, csv(dir.toFile()).map { it.readBytes() })
    }

    // ------------------------------------------------------------ регрессии после правок

    @Test
    fun `readCompactDataset counts classes only over rows actually loaded`(@TempDir dir: Path) {
        val dataDir = dir.toFile()
        fill(dataDir, rows = 5)
        // Нужное число колонок, но значения нечисловые: строка проходит быстрый подсчёт, но не парсится.
        File(dataDir, "legit_vectors.csv").appendText(List(343) { "x" }.joinToString(",") + "\n")

        val dataset = DatasetManager(dataDir, 16).readCompactDataset()

        assertEquals(10, dataset.sampleCount)
        assertEquals(5, dataset.legitCount) // раньше хвост массива меток давал 6
        assertEquals(5, dataset.cheatCount)
    }

    @Test
    fun `failed flush keeps vectors queued and retries them`(@TempDir dir: Path) {
        val dataDir = dir.toFile()
        val dm = DatasetManager(dataDir, 16)
        val legit = File(dataDir, "legit_vectors.csv")
        val original = legit.readBytes()
        legit.delete()
        assertTrue(legit.mkdir()) // дописать в каталог нельзя: appendText бросит IOException
        dm.enqueueVectorSample("legit", vector(1), "s", "h")

        assertFailsWith<java.io.IOException> { dm.flushPendingWrites() }
        assertEquals(1, dm.pendingWriteCount(), "вектор не должен пропасть")

        legit.delete()
        legit.writeBytes(original)
        assertEquals(1, dm.flushPendingWrites())
        assertEquals(0, dm.pendingWriteCount())
        assertEquals(1, dm.getDatasetSummary().legitVectors)
    }

    @Test
    fun `second archive operation is rejected while one is running`(@TempDir dir: Path) {
        val dataDir = dir.toFile()
        fill(dataDir, rows = 3)
        val dm = DatasetManager(dataDir, 16)
        val archive = File(dataDir, "dataset.dac")

        dm.runExclusiveArchiveOp {
            assertTrue(dm.isArchiveBusy())
            assertFailsWith<IllegalStateException> { DatasetArchive(dm).export(archive) }
            assertFailsWith<IllegalStateException> { DatasetArchive(dm).import(archive) }
        }
        assertTrue(!dm.isArchiveBusy())

        // Упавшая операция тоже обязана освободить флаг.
        assertFailsWith<Exception> { DatasetArchive(dm).import(File(dataDir, "missing.dac")) }
        assertTrue(!dm.isArchiveBusy())

        DatasetArchive(dm).export(archive)
        assertTrue(archive.exists())
    }

    @Test
    fun `bytes appended after the archive are rejected`(@TempDir dir: Path) {
        fill(dir.toFile(), rows = 5)
        val archive = File(dir.toFile(), "dataset.dac")
        val dm = DatasetManager(dir.toFile(), 16)
        DatasetArchive(dm).export(archive)
        archive.appendBytes(byteArrayOf(1, 2, 3))
        assertFailsWith<Exception> { DatasetArchive(dm).verify(archive) }
    }

    @Test
    fun `salt that differs only by trailing whitespace is the same salt`(@TempDir source: Path, @TempDir target: Path) {
        fill(source.toFile(), rows = 5)
        val archive = File(source.toFile(), "dataset.dac")
        DatasetArchive(DatasetManager(source.toFile(), 16)).export(archive)

        val targetDir = target.toFile()
        val targetDm = DatasetManager(targetDir, 16)
        File(targetDir, "dataset.salt").writeText(File(source.toFile(), "dataset.salt").readText().trim() + "\n")

        val result = DatasetArchive(targetDm).import(archive) // без --replace-salt: соли совпадают
        assertTrue(!result.saltReplaced)
    }

    @Test
    fun `replace-salt flag with identical salts replaces nothing`(@TempDir source: Path, @TempDir target: Path) {
        fill(source.toFile(), rows = 5)
        val archive = File(source.toFile(), "dataset.dac")
        DatasetArchive(DatasetManager(source.toFile(), 16)).export(archive)

        val targetDir = target.toFile()
        val targetDm = DatasetManager(targetDir, 16)
        File(targetDir, "dataset.salt").writeText(File(source.toFile(), "dataset.salt").readText())

        val result = DatasetArchive(targetDm).import(archive, DatasetArchive.ImportOptions(replaceSalt = true))
        assertTrue(!result.saltReplaced)
        assertTrue(result.backups.none { it.name.startsWith("dataset.salt") })
    }

    @Test
    fun `export leaves no temporary files behind`(@TempDir dir: Path) {
        fill(dir.toFile(), rows = 5)
        DatasetArchive(DatasetManager(dir.toFile(), 16)).export(File(dir.toFile(), "dataset.dac"))
        val leftovers = dir.toFile().list().orEmpty().filter { it.endsWith(".tmp") || it.startsWith(".dac-") }
        assertEquals(emptyList<String>(), leftovers)
    }

    @Test
    fun `export treats a missing csv as empty`(@TempDir dir: Path) {
        fill(dir.toFile(), rows = 5)
        val dm = DatasetManager(dir.toFile(), 16) // создаём до удаления: конструктор пересоздаёт файлы
        File(dir.toFile(), "cheat_vectors.csv").delete()

        val archive = File(dir.toFile(), "dataset.dac")
        DatasetArchive(dm).export(archive)
        val verified = DatasetArchive(dm).verify(archive)

        assertEquals(5, verified.legitRows)
        assertEquals(0, verified.cheatRows)
    }

    @Test
    fun `cli keeps flags out of positional arguments`() {
        val export = parseCliArgs(listOf("run/plugins/DaynAC", "--no-salt"))
        assertEquals(listOf("run/plugins/DaynAC"), export.positional)
        assertTrue("--no-salt" in export.flags)

        val import = parseCliArgs(listOf("--replace-salt", "a.dac", "run/plugins/DaynAC"))
        assertEquals(listOf("a.dac", "run/plugins/DaynAC"), import.positional)
        assertEquals(setOf("--replace-salt"), import.flags)
    }
}