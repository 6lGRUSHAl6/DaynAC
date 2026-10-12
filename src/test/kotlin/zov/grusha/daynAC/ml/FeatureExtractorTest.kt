package zov.grusha.daynAC.ml

import zov.grusha.daynAC.tracking.HitData
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class FeatureExtractorTest {

    private val extractor = FeatureExtractor(windowSize = 16)

    private fun hit(
        aimAngle: Double? = 5.0,
        distance: Double = 3.0,
        reachDistance: Double = 2.5,
        hitTimeDelta: Long? = 500L,
        hitTimeCV: Double? = 0.1,
        yawEntropyAbs: Double? = 1.5,
        pitchEntropyAbs: Double? = 1.0,
        yawEntropySigned: Double? = 1.8,
        pitchEntropySigned: Double? = 1.2,
        yawJitterAbs: Double? = 0.5,
        pitchJitterAbs: Double? = 0.3,
        yawJitterSigned: Double? = 0.4,
        pitchJitterSigned: Double? = 0.2,
        speedXZ: Float? = 0.15f,
        rotSmoothYaw: Double? = 2.0,
        rotSmoothPitch: Double? = 1.5,
        snapFactor: Double? = 0.1,
        microAdjustYaw: Int? = 3,
        microAdjustPitch: Int? = 2,
        jerkValue: Double? = 0.05,
        straightLineRatio: Double? = 0.3
    ): HitData = HitData(
        timestamp = 0L,
        aimAngle = aimAngle,
        distance = distance,
        reachDistance = reachDistance,
        hitTimeDelta = hitTimeDelta,
        hitTimeCV = hitTimeCV,
        yawEntropyAbs = yawEntropyAbs,
        pitchEntropyAbs = pitchEntropyAbs,
        yawEntropySigned = yawEntropySigned,
        pitchEntropySigned = pitchEntropySigned,
        yawJitterAbs = yawJitterAbs,
        pitchJitterAbs = pitchJitterAbs,
        yawJitterSigned = yawJitterSigned,
        pitchJitterSigned = pitchJitterSigned,
        speedXZ = speedXZ,
        rotSmoothYaw = rotSmoothYaw,
        rotSmoothPitch = rotSmoothPitch,
        snapFactor = snapFactor,
        microAdjustYaw = microAdjustYaw,
        microAdjustPitch = microAdjustPitch,
        jerkValue = jerkValue,
        straightLineRatio = straightLineRatio
    )

    // --- Инварианты структуры ------------------------------------------

    @Test
    fun `PER_HIT_NAMES has exactly PER_HIT_FEATURES elements`() {
        assertEquals(FeatureExtractor.PER_HIT_FEATURES, FeatureExtractor.PER_HIT_NAMES.size)
    }

    @Test
    fun `AGGREGATE_NAMES has exactly AGGREGATE_FEATURES elements`() {
        assertEquals(FeatureExtractor.AGGREGATE_FEATURES, FeatureExtractor.AGGREGATE_NAMES.size)
    }

    @Test
    fun `csvHeader has windowSize times PER_HIT plus AGGREGATE columns`() {
        val columns = FeatureExtractor.csvHeader(windowSize = 16).split(",")
        assertEquals(16 * FeatureExtractor.PER_HIT_FEATURES + FeatureExtractor.AGGREGATE_FEATURES, columns.size)
        assertEquals(343, columns.size)
    }

    // --- extract() ------------------------------------------------------

    @Test
    fun `extract returns null when history shorter than window`() {
        assertNull(extractor.extract(List(15) { hit() }))
        assertNull(extractor.extract(emptyList()))
    }

    @Test
    fun `extract returns vector of expected size`() {
        val vector = extractor.extract(List(16) { hit() })
        assertNotNull(vector)
        assertEquals(343, vector.size)
        assertEquals(16, vector.hitCount)
    }

    @Test
    fun `extract uses only last windowSize hits`() {
        val history = List(20) { i -> hit(aimAngle = i.toDouble()) }
        val vector = extractor.extract(history)!!
        assertEquals(4.0, vector.features[0], 1e-9)
        assertEquals(19.0, vector.features[15 * 21], 1e-9)
    }

    @Test
    fun `null aimAngle is sanitized to zero`() {
        val vector = extractor.extract(List(16) { hit(aimAngle = null) })!!
        assertEquals(0.0, vector.features[0], 1e-9)
        assertEquals(3.0, vector.features[1], 1e-9)
    }

    @Test
    fun `null hitTimeDelta is sanitized to zero`() {
        val vector = extractor.extract(List(16) { hit(hitTimeDelta = null) })!!
        assertEquals(0.0, vector.features[2], 1e-9)
    }

    @Test
    fun `NaN and Infinity are sanitized to zero`() {
        val vector = extractor.extract(List(16) { hit(aimAngle = Double.NaN, distance = Double.POSITIVE_INFINITY) })!!
        assertEquals(0.0, vector.features[0], 1e-9)
        assertEquals(0.0, vector.features[1], 1e-9)
    }

    // --- normalizeReach через extract() ---------------------------------

    @Test
    fun `reach at vanilla threshold maps to zero`() {
        val vector = extractor.extract(List(16) { hit(reachDistance = 3.0) })!!
        assertEquals(0.0, vector.features[20], 1e-9)
    }

    @Test
    fun `reach below threshold is negative and clamped`() {
        val zero = extractor.extract(List(16) { hit(reachDistance = 0.0) })!!
        assertEquals(-1.0, zero.features[20], 1e-9)

        val mid = extractor.extract(List(16) { hit(reachDistance = 1.5) })!!
        assertEquals(-0.5, mid.features[20], 1e-9)
    }

    @Test
    fun `reach above threshold is positive and clamped at one`() {
        val mid = extractor.extract(List(16) { hit(reachDistance = 4.5) })!!
        assertEquals(0.5, mid.features[20], 1e-9)

        val high = extractor.extract(List(16) { hit(reachDistance = 100.0) })!!
        assertEquals(1.0, high.features[20], 1e-9)
    }

    // --- FeatureVector.toCsvRow() ---------------------------------------

    @Test
    fun `toCsvRow produces featureCount columns`() {
        val vector = extractor.extract(List(16) { hit() })!!
        val row = vector.toCsvRow().split(",")
        assertEquals(343, row.size)
    }

    @Test
    fun `toCsvRow emits 0_0 for non-finite values`() {
        val features = DoubleArray(343) { Double.NaN }
        val vector = FeatureExtractor.FeatureVector(features, 16)
        assertTrue(vector.toCsvRow().split(",").all { it == "0.0" })
    }

    @Test
    fun `FeatureVector equality compares contents not references`() {
        val a = FeatureExtractor.FeatureVector(DoubleArray(343) { 1.0 }, 16)
        val b = FeatureExtractor.FeatureVector(DoubleArray(343) { 1.0 }, 16)
        assertEquals(a, b)
        assertEquals(a.hashCode(), b.hashCode())
    }
}