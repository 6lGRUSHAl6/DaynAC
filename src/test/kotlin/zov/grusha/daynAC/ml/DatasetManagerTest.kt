package zov.grusha.daynAC.ml

import org.junit.jupiter.api.io.TempDir
import java.io.File
import java.nio.file.Path
import java.util.UUID
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

class DatasetManagerTest {

    private fun vec(value: Double = 1.0): FeatureExtractor.FeatureVector =
        FeatureExtractor.FeatureVector(DoubleArray(343) { value }, 16)

    @Test
    fun `init creates both vector csv files with headers`(@TempDir tmp: Path) {
        DatasetManager(tmp.toFile(), windowSize = 16)
        assertTrue(File(tmp.toFile(), "legit_vectors.csv").exists())
        assertTrue(File(tmp.toFile(), "cheat_vectors.csv").exists())
        val header = File(tmp.toFile(), "legit_vectors.csv").readLines().first()
        assertEquals(345, header.split(",").size)
    }

    @Test
    fun `init does not create dead legit and cheat csv`(@TempDir tmp: Path) {
        DatasetManager(tmp.toFile(), windowSize = 16)
        assertTrue(!File(tmp.toFile(), "legit.csv").exists(), "legit.csv should not be created")
        assertTrue(!File(tmp.toFile(), "cheat.csv").exists(), "cheat.csv should not be created")
    }

    @Test
    fun `playerHash is stable for same UUID`(@TempDir tmp: Path) {
        val dm1 = DatasetManager(tmp.toFile(), windowSize = 16)
        val dm2 = DatasetManager(tmp.toFile(), windowSize = 16)
        val uuid = UUID.fromString("12345678-1234-1234-1234-123456789012")
        assertEquals(dm1.playerHash(uuid), dm2.playerHash(uuid))
    }

    @Test
    fun `playerHash differs for different UUIDs`(@TempDir tmp: Path) {
        val dm = DatasetManager(tmp.toFile(), windowSize = 16)
        assertNotEquals(dm.playerHash(UUID.randomUUID()), dm.playerHash(UUID.randomUUID()))
    }

    @Test
    fun `playerHash is not plain UUID`(@TempDir tmp: Path) {
        val dm = DatasetManager(tmp.toFile(), windowSize = 16)
        val uuid = UUID.randomUUID()
        val hash = dm.playerHash(uuid)
        assertTrue(!hash.contains(uuid.toString()), "hash must not leak UUID")
        assertEquals(12, hash.length)
    }

    @Test
    fun `writeVectorSample writes to legit file`(@TempDir tmp: Path) {
        val dm = DatasetManager(tmp.toFile(), windowSize = 16)
        dm.writeVectorSample("legit", vec(2.0), "session-1", "hash-1")

        val rows = File(tmp.toFile(), "legit_vectors.csv").readLines()
            .filter { it.isNotBlank() && !it.startsWith("h0") }
        assertEquals(1, rows.size)
        assertTrue(rows[0].startsWith("2.0,2.0"))
        assertTrue(rows[0].endsWith(",session-1,hash-1"))
    }

    @Test
    fun `writeVectorSample with unknown label is a no-op`(@TempDir tmp: Path) {
        val dm = DatasetManager(tmp.toFile(), windowSize = 16)
        dm.writeVectorSample("unknown", vec(), "s", "h")

        val (legit, cheat) = dm.getVectorDatasetStats()
        assertEquals(0, legit)
        assertEquals(0, cheat)
    }

    @Test
    fun `readVectorSamples reads back what was written`(@TempDir tmp: Path) {
        val dm = DatasetManager(tmp.toFile(), windowSize = 16)
        dm.writeVectorSample("legit", vec(0.5), "s1", "h1")
        dm.writeVectorSample("legit", vec(0.6), "s2", "h1")
        dm.writeVectorSample("cheat", vec(0.9), "s3", "h2")

        val dataset = dm.readVectorSamples()
        assertEquals(2, dataset.legitCount)
        assertEquals(1, dataset.cheatCount)
        assertEquals(3, dataset.samples.size)
        assertEquals(343, dataset.samples[0].features.size)
        assertEquals("s1", dataset.samples[0].sessionId)
        assertEquals("h1", dataset.samples[0].playerHash)
    }

    @Test
    fun `writeVectorSample escapes commas in sessionId`(@TempDir tmp: Path) {
        val dm = DatasetManager(tmp.toFile(), windowSize = 16)
        dm.writeVectorSample("legit", vec(), "session,with,commas", "h")

        val dataset = dm.readVectorSamples()
        assertEquals(1, dataset.legitCount)
        assertEquals("session_with_commas", dataset.samples[0].sessionId)
    }

    @Test
    fun `writeVectorSample escapes newlines in metadata`(@TempDir tmp: Path) {
        val dm = DatasetManager(tmp.toFile(), windowSize = 16)
        dm.writeVectorSample("legit", vec(), "s\r\n1", "h\r\n2")

        val dataset = dm.readVectorSamples()
        assertEquals(1, dataset.legitCount)
        assertEquals("s__1", dataset.samples[0].sessionId)
    }

    @Test
    fun `getVectorDatasetStats counts valid rows`(@TempDir tmp: Path) {
        val dm = DatasetManager(tmp.toFile(), windowSize = 16)
        repeat(3) { dm.writeVectorSample("legit", vec(), "s$it", "h") }
        repeat(5) { dm.writeVectorSample("cheat", vec(), "s$it", "h") }

        val (legit, cheat) = dm.getVectorDatasetStats()
        assertEquals(3, legit)
        assertEquals(5, cheat)
    }

    @Test
    fun `getSessionStats counts unique sessions per class`(@TempDir tmp: Path) {
        val dm = DatasetManager(tmp.toFile(), windowSize = 16)
        dm.writeVectorSample("legit", vec(), "s1", "h")
        dm.writeVectorSample("legit", vec(), "s1", "h")
        dm.writeVectorSample("legit", vec(), "s2", "h")
        dm.writeVectorSample("cheat", vec(), "s3", "h")

        val (legitSessions, cheatSessions) = dm.getSessionStats()
        assertEquals(2, legitSessions)
        assertEquals(1, cheatSessions)
    }

    @Test
    fun `readVectorSamples ignores header and blank lines`(@TempDir tmp: Path) {
        val dm = DatasetManager(tmp.toFile(), windowSize = 16)
        dm.writeVectorSample("legit", vec(), "s1", "h")

        File(tmp.toFile(), "legit_vectors.csv").appendText("\n")

        val dataset = dm.readVectorSamples()
        assertEquals(1, dataset.legitCount)
    }

    @Test
    fun `readVectorSamples skips rows with wrong column count`(@TempDir tmp: Path) {
        val dm = DatasetManager(tmp.toFile(), windowSize = 16)
        dm.writeVectorSample("legit", vec(), "s1", "h")

        File(tmp.toFile(), "legit_vectors.csv").appendText("1.0,2.0,3.0\n")

        val dataset = dm.readVectorSamples()
        assertEquals(1, dataset.legitCount)
    }

    @Test
    fun `salt file is created and reused`(@TempDir tmp: Path) {
        DatasetManager(tmp.toFile(), windowSize = 16)
        val saltFile = File(tmp.toFile(), "dataset.salt")
        assertTrue(saltFile.exists())
        val salt1 = saltFile.readText()

        DatasetManager(tmp.toFile(), windowSize = 16)
        assertEquals(salt1, saltFile.readText())
    }

    @Test
    fun `featureVersion file is created`(@TempDir tmp: Path) {
        DatasetManager(tmp.toFile(), windowSize = 16)
        val versionFile = File(tmp.toFile(), "dataset.featver")
        assertTrue(versionFile.exists())
        assertEquals(FeatureExtractor.FEATURE_VERSION.toString(), versionFile.readText().trim())
    }
}