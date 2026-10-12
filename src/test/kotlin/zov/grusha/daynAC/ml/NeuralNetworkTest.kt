package zov.grusha.daynAC.ml

import org.junit.jupiter.api.io.TempDir
import java.io.File
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class NeuralNetworkTest {

    private fun tinyNetwork() = NeuralNetwork(inputSize = 3, hiddenSizes = intArrayOf(4))

    private fun tinyDataset(): Pair<List<DoubleArray>, List<Double>> {
        val legit = List(5) { DoubleArray(3) { 0.1 } }
        val cheat = List(5) { DoubleArray(3) { 0.9 } }
        val labels = List(5) { 0.0 } + List(5) { 1.0 }
        return Pair(legit + cheat, labels)
    }

    // --- predict() -----------------------------------------------------

    @Test
    fun `predict returns value in zero-one range`() {
        val nn = tinyNetwork()
        val p = nn.predict(doubleArrayOf(0.5, 0.5, 0.5))
        assertTrue(p in 0.0..1.0, "p = $p")
    }

    @Test
    fun `predict throws on wrong input size`() {
        val nn = tinyNetwork()
        assertFailsWith<IllegalArgumentException> { nn.predict(doubleArrayOf(1.0, 2.0)) }
        assertFailsWith<IllegalArgumentException> { nn.predict(doubleArrayOf(1.0, 2.0, 3.0, 4.0)) }
    }

    @Test
    fun `predict is deterministic for same input`() {
        val nn = tinyNetwork()
        val input = doubleArrayOf(0.3, 0.7, 0.5)
        assertEquals(nn.predict(input), nn.predict(input), 1e-12)
    }

    // --- isCompatibleWith() --------------------------------------------

    @Test
    fun `isCompatibleWith returns false for missing file`() {
        val nn = tinyNetwork()
        assertFalse(nn.isCompatibleWith(File("definitely-not-here.dat")))
    }

    @Test
    fun `isCompatibleWith returns true after saveTo`(@TempDir tmp: Path) {
        val file = tmp.resolve("model.dat").toFile()
        val nn = tinyNetwork()
        nn.saveTo(file)
        assertTrue(nn.isCompatibleWith(file))
    }

    @Test
    fun `isCompatibleWith returns false after input size change`(@TempDir tmp: Path) {
        val file = tmp.resolve("model.dat").toFile()
        tinyNetwork().saveTo(file)
        val different = NeuralNetwork(inputSize = 5, hiddenSizes = intArrayOf(4))
        assertFalse(different.isCompatibleWith(file))
    }

    @Test
    fun `isCompatibleWith returns false on garbage file`(@TempDir tmp: Path) {
        val file = tmp.resolve("garbage.dat").toFile()
        file.writeText("this is not a model")
        assertFalse(tinyNetwork().isCompatibleWith(file))
    }

    // --- save / load roundtrip -----------------------------------------

    @Test
    fun `save and load preserve predictions after training`(@TempDir tmp: Path) {
        val file = tmp.resolve("model.dat").toFile()

        val nn1 = tinyNetwork()
        val (samples, labels) = tinyDataset()
        nn1.train(samples, labels, maxEpochs = 20, patience = 5)
        nn1.saveTo(file)

        val nn2 = tinyNetwork()
        nn2.loadFrom(file)

        for (probe in listOf(
            doubleArrayOf(0.1, 0.1, 0.1),
            doubleArrayOf(0.9, 0.9, 0.9),
            doubleArrayOf(0.5, 0.5, 0.5),
            doubleArrayOf(100.0, -50.0, 0.0)
        )) {
            assertEquals(nn1.predict(probe), nn2.predict(probe), 1e-9, "probe = ${probe.toList()}")
        }
    }

    @Test
    fun `loadFrom throws on garbage file`(@TempDir tmp: Path) {
        val file = tmp.resolve("bad.dat").toFile()
        file.writeText("not a model")
        assertFailsWith<IllegalStateException> { tinyNetwork().loadFrom(file) }
    }

    @Test
    fun `saveTo does not leave tmp file on success`(@TempDir tmp: Path) {
        val file = tmp.resolve("model.dat").toFile()
        tinyNetwork().saveTo(file)
        val leftover = File(file.parentFile, "model.dat.tmp")
        assertFalse(leftover.exists(), "temp file left behind: ${leftover.absolutePath}")
    }

    // --- train() валидация ---------------------------------------------

    @Test
    fun `train throws when samples and labels have different sizes`() {
        val nn = tinyNetwork()
        assertFailsWith<IllegalArgumentException> {
            nn.train(
                samples = List(10) { DoubleArray(3) },
                labels = List(9) { 0.0 },
                maxEpochs = 5
            )
        }
    }

    @Test
    fun `train throws when dataset is too small`() {
        val nn = tinyNetwork()
        assertFailsWith<IllegalArgumentException> {
            nn.train(
                samples = List(8) { DoubleArray(3) },
                labels = List(4) { 0.0 } + List(4) { 1.0 },
                maxEpochs = 5
            )
        }
    }

    @Test
    fun `train throws when one class has too few samples`() {
        val nn = tinyNetwork()
        assertFailsWith<IllegalArgumentException> {
            nn.train(
                samples = List(10) { DoubleArray(3) },
                labels = List(9) { 0.0 } + List(1) { 1.0 },
                maxEpochs = 5
            )
        }
    }

    @Test
    fun `train throws when a sample has wrong input size`() {
        val nn = tinyNetwork()
        assertFailsWith<IllegalArgumentException> {
            nn.train(
                samples = List(10) { if (it == 0) DoubleArray(5) else DoubleArray(3) },
                labels = List(5) { 0.0 } + List(5) { 1.0 },
                maxEpochs = 5
            )
        }
    }

    // --- train() результат ---------------------------------------------

    @Test
    fun `train returns TrainResult with finite metrics`() {
        val nn = tinyNetwork()
        val (samples, labels) = tinyDataset()
        val result = nn.train(samples, labels, maxEpochs = 20, patience = 5)

        assertTrue(result.epochsRun >= 1)
        assertTrue(result.trainLoss.isFinite())
        assertTrue(result.valLoss.isFinite())
        assertTrue(result.trainAccuracy in 0.0..1.0)
        assertTrue(result.valAccuracy in 0.0..1.0)
    }

    @Test
    fun `train returns SplitSummary covering all samples`() {
        val nn = tinyNetwork()
        val (samples, labels) = tinyDataset()
        val result = nn.train(samples, labels, maxEpochs = 20, patience = 5)

        val split = result.split
        assertNotNull(split)
        assertEquals(samples.size, split.trainSize + split.valSize + split.testSize)
    }

    @Test
    fun `train with groups summarizes as grouped when enough groups`() {
        val nn = tinyNetwork()
        val samples = List(15) { DoubleArray(3) { 0.0 } } + List(15) { DoubleArray(3) { 1.0 } }
        val labels = List(15) { 0.0 } + List(15) { 1.0 }
        val groups = List(15) { "legit-${it % 3}" } + List(15) { "cheat-${it % 3}" }

        val result = nn.train(samples, labels, groups = groups, maxEpochs = 10, patience = 5)
        assertTrue(result.split!!.fullyGrouped, "expected grouped split, got $result.split")
    }

    @Test
    fun `train with too few groups falls back to random split`() {
        val nn = tinyNetwork()
        val samples = List(10) { DoubleArray(3) } + List(10) { DoubleArray(3) }
        val labels = List(10) { 0.0 } + List(10) { 1.0 }
        val groups = List(10) { "legit-only" } + List(10) { "cheat-only" }

        val result = nn.train(samples, labels, groups = groups, maxEpochs = 10, patience = 5)
        assertFalse(result.split!!.fullyGrouped)
    }

    // --- evaluateThresholds() ------------------------------------------

    @Test
    fun `evaluateThresholds returns null before any training`() {
        val nn = tinyNetwork()
        assertEquals(null, nn.evaluateThresholds(doubleArrayOf(0.5)))
    }

    @Test
    fun `evaluateThresholds returns metrics after training`() {
        val nn = tinyNetwork()
        val (samples, labels) = tinyDataset()
        nn.train(samples, labels, maxEpochs = 20, patience = 5)

        val metrics = nn.evaluateThresholds(doubleArrayOf(0.3, 0.5, 0.7))
        assertNotNull(metrics)
        assertEquals(3, metrics.size)

        // evaluateThresholds работает на TEST-подмножестве, а не на всём датасете:
        // train() откладывает TEST_FRACTION (20%) в test-сет. При 5+5 это 1+1 = 2 окна.
        val totalInTest = metrics.first().totalLegit + metrics.first().totalCheat
        assertTrue(totalInTest in 2..10, "test set size = $totalInTest")

        for (m in metrics) {
            // Все пороги оцениваются на одном и том же test-сете
            assertEquals(totalInTest, m.totalLegit + m.totalCheat,
                "all thresholds must use the same test set")
            // Матрица ошибок должна быть согласована
            assertEquals(m.totalLegit, m.trueNegatives + m.falsePositives)
            assertEquals(m.totalCheat, m.truePositives + m.falseNegatives)
        }
    }

    @Test
    fun `ThresholdMetrics rates are in zero-one range`() {
        val nn = tinyNetwork()
        val (samples, labels) = tinyDataset()
        nn.train(samples, labels, maxEpochs = 20, patience = 5)

        val metrics = nn.evaluateThresholds(doubleArrayOf(0.5))!!.first()
        assertTrue(metrics.precision in 0.0..1.0)
        assertTrue(metrics.recall in 0.0..1.0)
        assertTrue(metrics.f1 in 0.0..1.0)
        assertTrue(metrics.falsePositiveRate in 0.0..1.0)
        assertTrue(metrics.falseNegativeRate in 0.0..1.0)
    }
}