package zov.grusha.daynAC.ml

import zov.grusha.daynAC.tracking.HitData
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertNull

/**
 * Структура вектора и контракт конструктора FeatureExtractor: валидация windowSize,
 * согласованность констант с именами признаков и CSV-заголовком, нормировка reach.
 */
class FeatureExtractorStructureTest {

    private fun hit(reachDistance: Double = 2.5): HitData = HitData(
        timestamp = 0L,
        aimAngle = 5.0,
        distance = 3.0,
        reachDistance = reachDistance,
        hitTimeDelta = 500L,
        hitTimeCV = 0.1,
        yawEntropyAbs = 1.5,
        pitchEntropyAbs = 1.0,
        yawEntropySigned = 1.8,
        pitchEntropySigned = 1.2,
        yawJitterAbs = 0.5,
        pitchJitterAbs = 0.3,
        yawJitterSigned = 0.4,
        pitchJitterSigned = 0.2,
        speedXZ = 0.15f,
        rotSmoothYaw = 2.0,
        rotSmoothPitch = 1.5,
        snapFactor = 0.1,
        microAdjustYaw = 3,
        microAdjustPitch = 2,
        jerkValue = 0.05,
        straightLineRatio = 0.3
    )

    // --- Валидация windowSize ---------------------------------------------

    @Test
    fun `zero window size is rejected at construction`() {
        assertFailsWith<IllegalArgumentException> { FeatureExtractor(windowSize = 0) }
    }

    @Test
    fun `negative window size is rejected at construction`() {
        assertFailsWith<IllegalArgumentException> { FeatureExtractor(windowSize = -3) }
    }

    @Test
    fun `window of one hit is accepted and gives a full-size vector`() {
        val extractor = FeatureExtractor(windowSize = 1)
        val vector = assertNotNull(extractor.extract(listOf(hit())))
        assertEquals(1 * FeatureExtractor.PER_HIT_FEATURES + FeatureExtractor.AGGREGATE_FEATURES, vector.features.size)
    }

    @Test
    fun `extract returns null when history is shorter than the window`() {
        val extractor = FeatureExtractor(windowSize = 4)
        assertNull(extractor.extract(List(3) { hit() }))
    }

    // --- Константы, имена и заголовок -------------------------------------

    @Test
    fun `per-hit names match PER_HIT_FEATURES`() {
        assertEquals(FeatureExtractor.PER_HIT_FEATURES, FeatureExtractor.PER_HIT_NAMES.size)
    }

    @Test
    fun `aggregate names match AGGREGATE_FEATURES`() {
        assertEquals(FeatureExtractor.AGGREGATE_FEATURES, FeatureExtractor.AGGREGATE_NAMES.size)
    }

    @Test
    fun `feature names are unique`() {
        assertEquals(FeatureExtractor.PER_HIT_NAMES.size, FeatureExtractor.PER_HIT_NAMES.distinct().size)
        assertEquals(FeatureExtractor.AGGREGATE_NAMES.size, FeatureExtractor.AGGREGATE_NAMES.distinct().size)
    }

    @Test
    fun `csv header has one column per vector element`() {
        for (windowSize in listOf(1, 4, 16)) {
            val extractor = FeatureExtractor(windowSize = windowSize)
            val vector = assertNotNull(extractor.extract(List(windowSize) { hit() }))
            val columns = FeatureExtractor.csvHeader(windowSize).split(",")

            assertEquals(vector.features.size, columns.size, "windowSize=$windowSize")
        }
    }

    @Test
    fun `default window gives 343 features`() {
        val extractor = FeatureExtractor()
        val vector = assertNotNull(extractor.extract(List(16) { hit() }))
        assertEquals(343, vector.features.size)
    }

    // --- reachNorm: нормированный reach -----------------------------------

    @Test
    fun `reach column is stored normalized, not in blocks`() {
        val extractor = FeatureExtractor(windowSize = 1)
        val reachIndex = FeatureExtractor.PER_HIT_NAMES.indexOf("reachNorm")
        assertEquals(FeatureExtractor.PER_HIT_FEATURES - 1, reachIndex)

        // (reach - 3) / 3, зажатое в [-1, 1].
        val cases = mapOf(
            0.0 to -1.0,
            1.5 to -0.5,
            3.0 to 0.0,
            4.5 to 0.5,
            6.0 to 1.0,
            20.0 to 1.0
        )
        for ((reach, expected) in cases) {
            val vector = assertNotNull(extractor.extract(listOf(hit(reachDistance = reach))))
            assertEquals(expected, vector.features[reachIndex], 1e-9, "reach=$reach")
        }
    }
}
