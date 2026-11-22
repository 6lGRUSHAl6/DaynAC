package zov.grusha.daynAC.ml

import java.util.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertContentEquals

class FastRandomTest {

    /** Сравниваем double побитово: 0.0 и -0.0 равны по ==, но это разные выходы RNG. */
    private fun bits(value: Double): Long = java.lang.Double.doubleToRawLongBits(value)

    @Test
    fun `FastRandom reproduces java util Random sequence bit for bit`() {
        val seeds = listOf(0L, 1L, 42L, -1L, Long.MIN_VALUE, Long.MAX_VALUE, 1234567890123L)

        for (seed in seeds) {
            val reference = Random(seed)
            val fast = NeuralNetwork.FastRandom(seed)

            repeat(2000) { i ->
                assertEquals(reference.nextInt(), fast.nextInt(), "nextInt, seed=$seed, draw=$i")
                assertEquals(
                    reference.nextInt(97), fast.nextInt(97),
                    "nextInt(97), seed=$seed, draw=$i"
                )
                assertEquals(reference.nextLong(), fast.nextLong(), "nextLong, seed=$seed, draw=$i")
                assertEquals(
                    reference.nextBoolean(), fast.nextBoolean(),
                    "nextBoolean, seed=$seed, draw=$i"
                )
                assertEquals(
                    bits(reference.nextDouble()), bits(fast.nextDouble()),
                    "nextDouble, seed=$seed, draw=$i"
                )
                assertEquals(
                    bits(reference.nextGaussian()), bits(fast.nextGaussian()),
                    "nextGaussian, seed=$seed, draw=$i"
                )
            }
        }
    }

    @Test
    fun `FastRandom produces the same shuffle as java util Random`() {
        for (seed in listOf(0L, 42L, -7L, Long.MAX_VALUE)) {
            val expected = (0 until 300).toMutableList()
            val actual = (0 until 300).toMutableList()

            java.util.Collections.shuffle(expected, Random(seed))
            java.util.Collections.shuffle(actual, NeuralNetwork.FastRandom(seed))

            assertContentEquals(expected.toIntArray(), actual.toIntArray(), "shuffle, seed=$seed")
        }
    }
}
