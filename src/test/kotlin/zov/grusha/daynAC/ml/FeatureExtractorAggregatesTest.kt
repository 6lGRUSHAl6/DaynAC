package zov.grusha.daynAC.ml

import zov.grusha.daynAC.tracking.HitData
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Агрегаты FeatureExtractor — последние [FeatureExtractor.AGGREGATE_FEATURES] чисел
 * вектора (индексы windowSize*PER_HIT_FEATURES .. +6). Функции приватные, поэтому
 * проверяются через extract() и хвост вектора.
 *
 * Null-агрегаты (недостаточно данных) проходят через sanitize() и попадают в вектор
 * как 0.0 — это тоже проверяется.
 */
class FeatureExtractorAggregatesTest {

    private val windowSize = 16
    private val extractor = FeatureExtractor(windowSize = windowSize)
    private val aggBase = windowSize * FeatureExtractor.PER_HIT_FEATURES

    private fun hit(
        aimAngle: Double? = 5.0,
        hitTimeDelta: Long? = 500L,
        distance: Double = 3.0,
        reachDistance: Double = 2.5,
        hitTimeCV: Double? = 0.1
    ): HitData = HitData(
        timestamp = 0L,
        aimAngle = aimAngle,
        distance = distance,
        reachDistance = reachDistance,
        hitTimeDelta = hitTimeDelta,
        hitTimeCV = hitTimeCV,
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

    private fun extractAggregates(history: List<HitData>): DoubleArray {
        val vector = extractor.extract(history)!!
        return DoubleArray(FeatureExtractor.AGGREGATE_FEATURES) { vector.features[aggBase + it] }
    }

    private fun agg(history: List<HitData>, index: Int): Double =
        extractAggregates(history)[index]

    // Индексы хвоста — по AGGREGATE_NAMES.
    private val meanAim = 0
    private val aimCv = 1
    private val hitDeltaCv = 2
    private val aimEntropy = 3
    private val aimAutocorr = 4
    private val timingAutocorr = 5
    private val trendR2 = 6

    // --- aggMeanAimAngle --------------------------------------------------

    @Test
    fun `mean aim angle is computed over the window`() {
        val history = List(windowSize) { hit(aimAngle = (it + 1).toDouble()) }
        assertEquals(8.5, agg(history, meanAim), 1e-9)
    }

    @Test
    fun `mean aim angle ignores null and non-finite values`() {
        val values = listOf(4.0, 6.0, null, Double.NaN, Double.POSITIVE_INFINITY) +
                List(windowSize - 5) { null }
        val history = values.map { hit(aimAngle = it) }

        assertEquals(5.0, agg(history, meanAim), 1e-9)
    }

    @Test
    fun `mean aim angle of only nulls is zero`() {
        val history = List(windowSize) { hit(aimAngle = null) }
        assertEquals(0.0, agg(history, meanAim), 1e-9)
    }

    // --- aggAimAngleCV ----------------------------------------------------

    @Test
    fun `aim angle CV is zero for identical angles`() {
        val history = List(windowSize) { hit(aimAngle = 12.0) }
        assertEquals(0.0, agg(history, aimCv), 1e-9)
    }

    @Test
    fun `aim angle CV matches known two-value sample`() {
        // [4, 6]: mean 5, выборочное std sqrt(2) -> CV sqrt(2)/5.
        val history = List(windowSize - 2) { hit(aimAngle = null) } +
                listOf(hit(aimAngle = 4.0), hit(aimAngle = 6.0))

        assertEquals(sqrt(2.0) / 5.0, agg(history, aimCv), 1e-9)
    }

    @Test
    fun `aim angle CV needs at least two values`() {
        val history = List(windowSize - 1) { hit(aimAngle = null) } + hit(aimAngle = 7.0)
        assertEquals(0.0, agg(history, aimCv), 1e-9)
    }

    // --- aggHitTimeDeltaCV ------------------------------------------------

    @Test
    fun `hit delta CV matches known two-value sample`() {
        // [100, 300]: mean 200, выборочное std sqrt(20000) -> CV sqrt(20000)/200.
        val history = List(windowSize - 2) { hit(hitTimeDelta = null) } +
                listOf(hit(hitTimeDelta = 100), hit(hitTimeDelta = 300))

        assertEquals(sqrt(20_000.0) / 200.0, agg(history, hitDeltaCv), 1e-9)
    }

    @Test
    fun `hit delta CV is zero for metronomic timing`() {
        val history = List(windowSize) { hit(hitTimeDelta = 100) }
        assertEquals(0.0, agg(history, hitDeltaCv), 1e-9)
    }

    @Test
    fun `hit delta CV ignores null deltas`() {
        val history = List(windowSize - 2) { hit(hitTimeDelta = null) } +
                listOf(hit(hitTimeDelta = 150), hit(hitTimeDelta = 150))
        assertEquals(0.0, agg(history, hitDeltaCv), 1e-9)
    }

    // --- aggAimAngleEntropy -----------------------------------------------

    @Test
    fun `aim angle entropy is zero when all angles share a bin`() {
        val history = List(windowSize) { hit(aimAngle = 5.0) } // 5/180*8 -> бин 0
        assertEquals(0.0, agg(history, aimEntropy), 1e-9)
    }

    @Test
    fun `aim angle entropy is three bits for uniform eight bins`() {
        // Центры восьми бинов на [0, 180]; два полных прохода по 16 ударам.
        val centers = (0 until 8).map { (it + 0.5) * 180.0 / 8.0 }
        val history = (centers + centers).map { hit(aimAngle = it) }

        assertEquals(3.0, agg(history, aimEntropy), 1e-9)
    }

    @Test
    fun `aim angle entropy clamps out-of-range angles into edge bins`() {
        // Отрицательный угол -> бин 0, больше 180 -> бин 7: два бина -> 1 бит.
        val history = List(windowSize / 2) { hit(aimAngle = -50.0) } +
                List(windowSize / 2) { hit(aimAngle = 500.0) }
        assertEquals(1.0, agg(history, aimEntropy), 1e-9)
    }

    // --- aggAimAngleAutocorrLag1 ------------------------------------------

    @Test
    fun `aim angle autocorrelation is one for arithmetic progression`() {
        val history = List(windowSize) { hit(aimAngle = (it + 1).toDouble()) }
        assertEquals(1.0, agg(history, aimAutocorr), 1e-9)
    }

    @Test
    fun `aim angle autocorrelation is one for constant series`() {
        // Нулевая дисперсия: корреляция формально 0/0, но каждое следующее значение
        // полностью предсказуемо — кодируем 1.0, а не 0.0 («шум без корреляции»).
        val history = List(windowSize) { hit(aimAngle = 5.0) }
        assertEquals(1.0, agg(history, aimAutocorr), 1e-9)
    }

    @Test
    fun `aim angle autocorrelation is one for constant series that is not exact in binary`() {
        // 0.1 не представимо точно: суммы накапливают ошибку, и без порога вместо «нулевой
        // дисперсии» вышло бы ±1e-20 (вплоть до NaN из sqrt отрицательного).
        val history = List(windowSize) { hit(aimAngle = 0.1) }
        assertEquals(1.0, agg(history, aimAutocorr), 1e-9)
    }

    @Test
    fun `aim angle autocorrelation is one for all-zero series`() {
        val history = List(windowSize) { hit(aimAngle = 0.0) }
        assertEquals(1.0, agg(history, aimAutocorr), 1e-9)
    }

    @Test
    fun `aim angle autocorrelation is zero when only one side is constant`() {
        // Пары (5,5), (5,5), (5,7): левая сторона константна, правая нет -> корреляция не определена.
        val history = List(windowSize - 4) { hit(aimAngle = null) } +
                listOf(5.0, 5.0, 5.0, 7.0).map { hit(aimAngle = it) }
        assertEquals(0.0, agg(history, aimAutocorr), 1e-9)
    }

    @Test
    fun `aim angle autocorrelation is zero when both sides are constant at different levels`() {
        // Пары (5,7) и (5,7): обе стороны константны, но это не «повтор значения».
        val history = List(windowSize - 5) { hit(aimAngle = null) } +
                listOf(5.0, 7.0, null, 5.0, 7.0).map { hit(aimAngle = it) }
        assertEquals(0.0, agg(history, aimAutocorr), 1e-9)
    }

    @Test
    fun `aim angle autocorrelation needs at least two adjacent pairs`() {
        // Два значения подряд -> одна пара.
        val history = List(windowSize - 2) { hit(aimAngle = null) } +
                listOf(hit(aimAngle = 1.0), hit(aimAngle = 2.0))
        assertEquals(0.0, agg(history, aimAutocorr), 1e-9)
    }

    @Test
    fun `aim angle autocorrelation is defined for three adjacent values`() {
        // [1, 2, 3] -> пары (1,2), (2,3): идеальная линейная связь.
        val history = List(windowSize - 3) { hit(aimAngle = null) } +
                listOf(1.0, 2.0, 3.0).map { hit(aimAngle = it) }
        assertEquals(1.0, agg(history, aimAutocorr), 1e-9)
    }

    @Test
    fun `aim angle autocorrelation ignores values that are not adjacent`() {
        // Три значения, но между ними пропуски: ни одной соседней пары.
        val history = List(windowSize) { hit(aimAngle = if (it % 5 == 0) it.toDouble() else null) }
        assertEquals(0.0, agg(history, aimAutocorr), 1e-9)
    }

    @Test
    fun `aim angle autocorrelation does not stitch values across a gap`() {
        // 1..8, разрыв (NaN), 1..7. Пары берутся внутри сегментов: везде y = x + 1 -> ровно 1.0.
        // Если бы ряд склеивался, пара (8, 1) через разрыв сильно снизила бы корреляцию.
        val angles = (1..8).map { it.toDouble() } + listOf(Double.NaN) + (1..7).map { it.toDouble() }
        val history = angles.map { hit(aimAngle = it) }
        assertEquals(windowSize, history.size)
        assertEquals(1.0, agg(history, aimAutocorr), 1e-9)
    }

    // --- aggHitTimingAutocorrLag1 -----------------------------------------

    @Test
    fun `hit timing autocorrelation is one for monotone timing`() {
        val history = List(windowSize) { hit(hitTimeDelta = (it + 1).toLong()) }
        assertEquals(1.0, agg(history, timingAutocorr), 1e-9)
    }

    @Test
    fun `hit timing autocorrelation is one for constant timing`() {
        // Метроном (таймер-чит с одинаковым интервалом в мс) — самый механический паттерн.
        val history = List(windowSize) { hit(hitTimeDelta = 200) }
        assertEquals(1.0, agg(history, timingAutocorr), 1e-9)
    }

    @Test
    fun `hit timing autocorrelation does not stitch series across a reset gap`() {
        // После resetHitTiming первый удар новой серии имеет hitTimeDelta = null.
        // 1..8, разрыв, 1..7: пары только внутри серий -> y = x + 1 -> 1.0.
        val deltas: List<Long?> = (1L..8L).toList() + listOf(null) + (1L..7L).toList()
        val history = deltas.map { hit(hitTimeDelta = it) }
        assertEquals(windowSize, history.size)
        assertEquals(1.0, agg(history, timingAutocorr), 1e-9)
    }

    // --- aggAimAngleTrendR2 -----------------------------------------------

    @Test
    fun `aim angle trend is one for linear convergence`() {
        val history = List(windowSize) { hit(aimAngle = 100.0 - it) }
        assertEquals(1.0, agg(history, trendR2), 1e-9)
    }

    @Test
    fun `aim angle trend is undefined for flat angles`() {
        // Константный угол не «сходится»: тренда нет -> 0.0 (а не 1.0).
        val history = List(windowSize) { hit(aimAngle = 10.0) }
        assertEquals(0.0, agg(history, trendR2), 1e-9)
    }

    @Test
    fun `aim angle trend is undefined for flat angles that are not exact in binary`() {
        // Ошибка округления сумм не должна давать «тренд» из шума размером 1e-34.
        val history = List(windowSize) { hit(aimAngle = 0.1) }
        assertEquals(0.0, agg(history, trendR2), 1e-9)
    }

    @Test
    fun `aim angle trend uses real hit positions when some angles are missing`() {
        // Линейное схождение 100 - 2*i с пропусками на позициях 3 и 7. По реальным позициям
        // это идеальная прямая (R2 = 1). По индексам отфильтрованного списка шаг был бы
        // то 2, то 4, и R2 вышел бы ниже единицы.
        val history = List(windowSize) { i ->
            hit(aimAngle = if (i == 3 || i == 7) null else 100.0 - 2.0 * i)
        }
        assertEquals(1.0, agg(history, trendR2), 1e-9)
    }

    @Test
    fun `aim angle trend is below one for alternating angles`() {
        val history = List(windowSize) { hit(aimAngle = if (it % 2 == 0) 5.0 else 50.0) }
        val r2 = agg(history, trendR2)
        assertTrue(r2 < 0.1, "симметричный зигзаг без наклона не должен давать линейный тренд, R2=$r2")
    }

    // --- Границы окна и санитизация ---------------------------------------

    @Test
    fun `aggregates are computed from the last windowSize hits only`() {
        val history = List(4) { hit(aimAngle = 180.0) } +
                List(windowSize) { hit(aimAngle = 10.0) }

        assertEquals(10.0, agg(history, meanAim), 1e-9)
        assertEquals(0.0, agg(history, aimCv), 1e-9)
    }

    @Test
    fun `all aggregates are zero when all inputs are unusable`() {
        val history = List(windowSize) { hit(aimAngle = null, hitTimeDelta = null) }
        val aggregates = extractAggregates(history)
        for ((index, value) in aggregates.withIndex()) {
            assertEquals(0.0, value, 1e-9, "агрегат $index")
        }
    }
}
