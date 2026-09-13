package zov.grusha.daynAC.ml

import zov.grusha.daynAC.tracking.HitData
import kotlin.math.log2
import kotlin.math.sqrt

/**
 * Преобразует историю последних ударов игрока (из [zov.grusha.daynAC.tracking.HitTracker])
 * в фиксированный вектор признаков для нейросети.
 *
 * Структура вектора (windowSize * 20 + 7 элементов, при windowSize=16 -> 327):
 *  - windowSize (16) последних ударов × 20 признаков на удар = 320
 *  - 7 агрегатных статистик по окну
 *
 * CPS (clicks per second) сознательно НЕ включён: проект целится в 1.9 no-cooldown
 * PvP, где легит-игроки тренируют высокий и стабильный CPS сами по себе, так что
 * абсолютная скорость кликов — слабый разделитель. Вместо неё за "механическую
 * регулярность" тайминга отвечает hitTimingAutocorr (предсказуемость интервалов,
 * а не их абсолютная величина).
 *
 * Вектор всегда одной длины независимо от состояния игры — это обязательное
 * требование для подачи в нейросеть. Если истории не хватает, extract()
 * возвращает null (недостаточно данных для осмысленного анализа) — без
 * zero-padding, чтобы не кормить сеть искусственными нулями на старте боя.
 *
 * Все null-поля HitData кодируются нулём — одинаково и при записи датасета,
 * и при inference, чтобы train и runtime видели одну и ту же картину.
 */
class FeatureExtractor(
    private val windowSize: Int = 16
) {

    data class FeatureVector(
        val features: DoubleArray,
        val hitCount: Int
    ) {
        val size: Int get() = features.size

        /** Строка для CSV (значения через запятую, без метки). */
        fun toCsvRow(): String = features.joinToString(",") { format(it) }

        override fun equals(other: Any?): Boolean =
            other is FeatureVector && features.contentEquals(other.features) && hitCount == other.hitCount

        override fun hashCode(): Int = features.contentHashCode() * 31 + hitCount

        companion object {
            private fun format(v: Double): String =
                if (v.isFinite()) v.toString() else "0.0"
        }
    }

    companion object {
        /** Сколько числовых признаков берётся из каждого HitData (timestamp не считается). */
        const val PER_HIT_FEATURES = 20

        /** Сколько агрегатных статистик считается по всему окну ударов. */
        const val AGGREGATE_FEATURES = 7

        /**
         * Порядок признаков внутри одного удара. ВАЖНО: этот порядок
         * фиксируется навсегда — модель обучается именно на нём.
         */
        val PER_HIT_NAMES: List<String> = listOf(
            "aimAngle",
            "distance",
            "hitTimeDelta",
            "hitTimeCV",
            "yawEntropyAbs",
            "pitchEntropyAbs",
            "yawEntropySigned",
            "pitchEntropySigned",
            "yawJitterAbs",
            "pitchJitterAbs",
            "yawJitterSigned",
            "pitchJitterSigned",
            "speedXZ",
            "rotSmoothYaw",
            "rotSmoothPitch",
            "snapFactor",
            "microAdjustYaw",
            "microAdjustPitch",
            "jerkValue",
            "straightLineRatio"
        )

        val AGGREGATE_NAMES: List<String> = listOf(
            "aggMeanAimAngle",         // средний угол прицеливания по окну
            "aggAimAngleCV",           // коэффициент вариации угла прицеливания
            "aggHitTimeDeltaCV",       // коэффициент вариации интервалов между ударами (по окну ударов)
            "aggAimAngleEntropy",      // энтропия Шеннона распределения углов по окну
            "aggAimAngleAutocorrLag1", // автокорреляция угла (lag-1): насколько следующий удар похож на предыдущий
            "aggHitTimingAutocorrLag1",// автокорреляция интервалов между ударами (lag-1): механическая регулярность
            "aggAimAngleTrendR2"       // R² линейного тренда угла: убывает ли угол к концу окна (дожатие прицела)
        )

        /** Полный заголовок CSV для вектора. */
        fun csvHeader(windowSize: Int): String =
            (0 until windowSize).flatMap { i ->
                PER_HIT_NAMES.map { name -> "h$i$name" }
            }.plus(AGGREGATE_NAMES).joinToString(",")
    }

    /**
     * Строит вектор признаков из истории ударов.
     *
     * @param history удары в хронологическом порядке (старые -> новые),
     *                как их отдаёт HitTracker.getHistory()
     * @return вектор длины windowSize * 20 + 7 или null, если ударов меньше windowSize
     */
    fun extract(history: List<HitData>): FeatureVector? {
        if (history.size < windowSize) return null

        // Берём последние windowSize ударов (самые свежие)
        val window = history.takeLast(windowSize)
        val out = DoubleArray(windowSize * PER_HIT_FEATURES + AGGREGATE_FEATURES)

        for ((hitIndex, hit) in window.withIndex()) {
            val base = hitIndex * PER_HIT_FEATURES
            out[base + 0] = sanitize(hit.aimAngle)
            out[base + 1] = sanitize(hit.distance)
            out[base + 2] = sanitize(hit.hitTimeDelta)
            out[base + 3] = sanitize(hit.hitTimeCV)
            out[base + 4] = sanitize(hit.yawEntropyAbs)
            out[base + 5] = sanitize(hit.pitchEntropyAbs)
            out[base + 6] = sanitize(hit.yawEntropySigned)
            out[base + 7] = sanitize(hit.pitchEntropySigned)
            out[base + 8] = sanitize(hit.yawJitterAbs)
            out[base + 9] = sanitize(hit.pitchJitterAbs)
            out[base + 10] = sanitize(hit.yawJitterSigned)
            out[base + 11] = sanitize(hit.pitchJitterSigned)
            out[base + 12] = sanitize(hit.speedXZ)
            out[base + 13] = sanitize(hit.rotSmoothYaw)
            out[base + 14] = sanitize(hit.rotSmoothPitch)
            out[base + 15] = sanitize(hit.snapFactor)
            out[base + 16] = sanitize(hit.microAdjustYaw)
            out[base + 17] = sanitize(hit.microAdjustPitch)
            out[base + 18] = sanitize(hit.jerkValue)
            out[base + 19] = sanitize(hit.straightLineRatio)
        }

        val aggBase = windowSize * PER_HIT_FEATURES
        out[aggBase + 0] = sanitize(meanAimAngle(window))
        out[aggBase + 1] = sanitize(aimAngleCV(window))
        out[aggBase + 2] = sanitize(hitTimeDeltaCV(window))
        out[aggBase + 3] = sanitize(aimAngleEntropy(window))
        out[aggBase + 4] = sanitize(aimAngleAutocorrLag1(window))
        out[aggBase + 5] = sanitize(hitTimingAutocorrLag1(window))
        out[aggBase + 6] = sanitize(aimAngleTrendR2(window))

        return FeatureVector(out, window.size)
    }

    // ------------------------------------------------------------------
    // Вспомогательное
    // ------------------------------------------------------------------

    /** Null и нечисловые значения (NaN/Infinity) превращаем в 0.0. */
    private fun sanitize(value: Double?): Double =
        value?.takeIf { it.isFinite() } ?: 0.0

    private fun sanitize(value: Float?): Double =
        value?.takeIf { it.isFinite() }?.toDouble() ?: 0.0

    private fun sanitize(value: Long?): Double =
        value?.toDouble() ?: 0.0

    private fun sanitize(value: Int?): Double =
        value?.toDouble() ?: 0.0

    /** Только конечные значения поля по окну (null пропускаем). */
    private fun finiteValues(window: List<HitData>, selector: (HitData) -> Double?): List<Double> =
        window.mapNotNull { hit ->
            selector(hit)?.takeIf { it.isFinite() }
        }

    private fun meanAimAngle(window: List<HitData>): Double? {
        val values = finiteValues(window) { it.aimAngle }
        if (values.isEmpty()) return null
        return values.sum() / values.size
    }

    /** Коэффициент вариации (stdDev / mean) угла прицеливания. */
    private fun aimAngleCV(window: List<HitData>): Double? {
        val values = finiteValues(window) { it.aimAngle }
        return coefficientOfVariation(values)
    }

    /** CV интервалов между ударами по окну — свой, отдельный от скользящего hitTimeCV из SnapshotTracker. */
    private fun hitTimeDeltaCV(window: List<HitData>): Double? {
        val values = finiteValues(window) { it.hitTimeDelta?.toDouble() }
        return coefficientOfVariation(values)
    }

    private fun coefficientOfVariation(values: List<Double>): Double? {
        if (values.size < 2) return null
        val mean = values.sum() / values.size
        if (mean <= 0.0) return null
        val variance = values.sumOf { val d = it - mean; d * d } / (values.size - 1)
        return sqrt(variance) / mean
    }

    /**
     * Энтропия Шеннона распределения углов прицеливания по 8 бинам в диапазоне [0, 180] градусов
     * (весь фактический диапазон CombatMath.getAimAngle). Ровный читерский аим
     * кучкуется в 1-2 бинах -> низкая энтропия; человеческий разбросан -> высокая.
     */
    private fun aimAngleEntropy(window: List<HitData>): Double? {
        val values = finiteValues(window) { it.aimAngle }
        if (values.size < 2) return null

        val binCount = 8
        val maxRange = 180.0
        val bins = IntArray(binCount)
        for (v in values) {
            val binIndex = ((v.coerceIn(0.0, maxRange) / maxRange) * binCount).toInt().coerceIn(0, binCount - 1)
            bins[binIndex]++
        }

        var entropy = 0.0
        for (count in bins) {
            if (count > 0) {
                val p = count.toDouble() / values.size
                entropy -= p * log2(p)
            }
        }
        return entropy
    }

    /**
     * Автокорреляция lag-1 произвольной числовой последовательности: корреляция
     * ряда с самим собой, сдвинутым на одну позицию. Высокое значение = следующий
     * элемент предсказуем из предыдущего (машинный/механический паттерн).
     * Используется и для угла прицеливания, и для интервалов между ударами —
     * логика идентична, различается только входная последовательность.
     */
    private fun autocorrLag1(values: List<Double>): Double? {
        if (values.size < 3) return null

        val n = values.size - 1 // число пар (x[i], x[i+1])
        var sumX = 0.0; var sumY = 0.0; var sumXY = 0.0
        var sumX2 = 0.0; var sumY2 = 0.0
        for (i in 0 until n) {
            val x = values[i]
            val y = values[i + 1]
            sumX += x; sumY += y
            sumXY += x * y
            sumX2 += x * x; sumY2 += y * y
        }

        val numerator = n * sumXY - sumX * sumY
        val denominator = sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY))
        if (denominator <= 0.0) return null
        return (numerator / denominator).coerceIn(-1.0, 1.0)
    }

    /** Автокорреляция lag-1 угла прицеливания: плавно и предсказуемо ли меняется аим от удара к удару. */
    private fun aimAngleAutocorrLag1(window: List<HitData>): Double? {
        val values = finiteValues(window) { it.aimAngle }
        return autocorrLag1(values)
    }

    /**
     * Автокорреляция lag-1 интервалов между ударами: насколько текущий hitTimeDelta
     * предсказывается предыдущим. Высокое значение — механическая регулярность
     * тайминга (например, таймер-чит), даже если сам по себе CPS не аномален.
     */
    private fun hitTimingAutocorrLag1(window: List<HitData>): Double? {
        val values = finiteValues(window) { it.hitTimeDelta?.toDouble() }
        return autocorrLag1(values)
    }

    /**
     * R² линейной регрессии угла прицеливания по номеру удара в окне.
     * R² ~ 1 означает монотонное схождение прицела (каждый удар точнее
     * предыдущего на одну и ту же величину) — так люди не наводятся.
     */
    private fun aimAngleTrendR2(window: List<HitData>): Double? {
        val values = finiteValues(window) { it.aimAngle }
        if (values.size < 3) return null

        val n = values.size
        var sumX = 0.0; var sumY = 0.0; var sumXY = 0.0; var sumX2 = 0.0
        for (i in 0 until n) {
            val x = i.toDouble()
            val y = values[i]
            sumX += x; sumY += y
            sumXY += x * y
            sumX2 += x * x
        }

        val meanX = sumX / n
        val meanY = sumY / n

        var ssTot = 0.0
        for (v in values) {
            val d = v - meanY
            ssTot += d * d
        }
        if (ssTot == 0.0) return null // все углы одинаковы -> тренд не определён

        val denominator = n * sumX2 - sumX * sumX
        if (denominator == 0.0) return null
        val b = (n * sumXY - sumX * sumY) / denominator
        val a = meanY - b * meanX

        var ssRes = 0.0
        for (i in 0 until n) {
            val diff = values[i] - (a + b * i)
            ssRes += diff * diff
        }

        return (1.0 - ssRes / ssTot).coerceIn(0.0, 1.0)
    }
}