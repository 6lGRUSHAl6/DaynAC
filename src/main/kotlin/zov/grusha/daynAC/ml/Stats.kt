package zov.grusha.daynAC.ml

import kotlin.math.ceil
import kotlin.math.sqrt

object Stats {
    const val Z_95 = 1.96

    fun wilsonInterval(successes: Int, total: Int, z: Double = Z_95): Pair<Double, Double> {
        require(total >= 0 && successes in 0..total)
        if (total == 0) return Pair(0.0, 1.0)

        val n = total.toDouble()
        val p = successes / n
        val z2 = z * z
        val denominator = 1.0 + z2 / n
        val center = (p + z2 / (2.0 * n)) / denominator
        val margin = z * sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denominator
        return Pair((center - margin).coerceAtLeast(0.0), (center + margin).coerceAtMost(1.0))
    }

    /** Сколько наблюдений нужно, чтобы 0 ошибок дали верхнюю границу ≤ targetUpper. */
    fun minSamplesForZeroErrors(targetUpper: Double, z: Double = Z_95): Int =
        ceil(z * z * (1.0 - targetUpper) / targetUpper).toInt()
}