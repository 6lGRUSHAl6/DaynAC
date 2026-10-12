package zov.grusha.daynAC.ml

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class StatsTest {

    @Test
    fun `wilson with zero successes has low at zero`() {
        val (low, high) = Stats.wilsonInterval(0, 100)
        assertEquals(0.0, low, 1e-9)
        assertTrue(high in 0.03..0.045, "high = $high")
    }

    @Test
    fun `wilson with all successes has high at one`() {
        val (low, high) = Stats.wilsonInterval(100, 100)
        assertEquals(1.0, high, 1e-9)
        assertTrue(low in 0.955..0.97, "low = $low")
    }

    @Test
    fun `wilson is centered on proportion for balanced sample`() {
        val (low, high) = Stats.wilsonInterval(50, 100)
        val center = (low + high) / 2.0
        assertEquals(0.5, center, 0.01)
        assertTrue(high - low in 0.15..0.25, "width = ${high - low}")
    }

    @Test
    fun `wilson for empty sample is widest possible`() {
        val (low, high) = Stats.wilsonInterval(0, 0)
        assertEquals(0.0, low, 1e-9)
        assertEquals(1.0, high, 1e-9)
    }

    @Test
    fun `wilson widens as sample shrinks`() {
        val wide = Stats.wilsonInterval(1, 10)
        val narrow = Stats.wilsonInterval(100, 1000)
        assertTrue((wide.second - wide.first) > (narrow.second - narrow.first))
    }

    @Test
    fun `wilson rejects invalid arguments`() {
        assertFailsWith<IllegalArgumentException> { Stats.wilsonInterval(-1, 10) }
        assertFailsWith<IllegalArgumentException> { Stats.wilsonInterval(11, 10) }
        assertFailsWith<IllegalArgumentException> { Stats.wilsonInterval(5, -1) }
    }

    @Test
    fun `minSamplesForZeroErrors with one percent target returns 381`() {
        assertEquals(381, Stats.minSamplesForZeroErrors(0.01))
    }

    @Test
    fun `minSamplesForZeroErrors scales with target`() {
        assertTrue(Stats.minSamplesForZeroErrors(0.001) > Stats.minSamplesForZeroErrors(0.01))
        assertTrue(Stats.minSamplesForZeroErrors(0.01) > Stats.minSamplesForZeroErrors(0.05))
    }
}