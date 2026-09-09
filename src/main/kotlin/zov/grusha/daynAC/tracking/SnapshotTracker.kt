package zov.grusha.daynAC.tracking

import org.bukkit.entity.Player
import java.util.ArrayDeque
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap
import kotlin.math.abs
import kotlin.math.log2
import kotlin.math.sqrt

class SnapshotTracker(private val maxSize: Int = 20) {

    private data class PlayerHeadState(
        var lastYaw: Float,
        var lastPitch: Float,
        var lastX: Double,
        var lastZ: Double,
        var lastYawDeltaAbs: Float = 0f,
        var lastPitchDeltaAbs: Float = 0f,
        var lastHitTime: Long? = null,
        val hitDeltas: LongArray = LongArray(10),
        var hitDeltasCount: Int = 0,
        var hitDeltasIndex: Int = 0
    )

    private val playerStates = ConcurrentHashMap<UUID, PlayerHeadState>()
    private val historySnapshot = ConcurrentHashMap<UUID, ArrayDeque<PlayerSnapshot>>()
    private val lastAimAngles = ConcurrentHashMap<UUID, Double>()

    private fun getAngleDeltaSigned(current: Float, previous: Float): Float {
        var delta = (current - previous) % 360f
        if (delta > 180f) delta -= 360f
        if (delta < -180f) delta += 360f
        return delta
    }

    /**
     * Регистрирует удар и возвращает пару: (интервал времени с предыдущего удара, предыдущий угол прицеливания).
     * Также сохраняет текущий угол как последний для будущих вызовов.
     */
    fun registerHitWithAngle(player: Player, currentAimAngle: Double?): Pair<Long?, Double?> {
        val uuid = player.uniqueId
        val now = System.currentTimeMillis()
        val loc = player.location

        val state = playerStates.computeIfAbsent(uuid) {
            PlayerHeadState(
                lastYaw = loc.yaw,
                lastPitch = loc.pitch,
                lastX = loc.x,
                lastZ = loc.z
            )
        }

        val delta = state.lastHitTime?.let { now - it }
        state.lastHitTime = now

        if (delta != null) {
            if (delta < 2000L) {
                state.hitDeltas[state.hitDeltasIndex] = delta
                state.hitDeltasIndex = (state.hitDeltasIndex + 1) % state.hitDeltas.size
                if (state.hitDeltasCount < state.hitDeltas.size) {
                    state.hitDeltasCount++
                }
            } else {
                state.hitDeltasCount = 0
                state.hitDeltasIndex = 0
            }
        }

        // Получаем предыдущий угол и сохраняем новый
        val prevAngle = lastAimAngles[uuid]
        if (currentAimAngle != null && currentAimAngle.isFinite() && currentAimAngle >= 0) {
            lastAimAngles[uuid] = currentAimAngle
        }
        return Pair(delta, prevAngle)
    }

    // Старый метод оставляем для совместимости, но он не используется в новой логике
    @Deprecated("Use registerHitWithAngle instead")
    fun registerHit(player: Player): Long? {
        return registerHitWithAngle(player, null).first
    }

    fun getHitTimeCV(player: Player): Double? {
        val state = playerStates[player.uniqueId] ?: return null
        val n = state.hitDeltasCount
        if (n < 2) return null

        var sum = 0.0
        for (i in 0 until n) {
            sum += state.hitDeltas[i]
        }

        val mean = sum / n
        if (mean <= 0.0) return null

        var varianceSum = 0.0
        for (i in 0 until n) {
            val diff = state.hitDeltas[i] - mean
            varianceSum += diff * diff
        }

        val stdDev = sqrt(varianceSum / (n - 1))
        return stdDev / mean
    }

    fun recordSnapshot(player: Player): PlayerSnapshot? {
        val uuid = player.uniqueId
        val currentYaw = player.location.yaw
        val currentPitch = player.location.pitch
        val currentX = player.location.x
        val currentZ = player.location.z
        val state = playerStates[uuid]

        if (state == null) {
            playerStates[uuid] = PlayerHeadState(
                lastYaw = currentYaw,
                lastPitch = currentPitch,
                lastX = currentX,
                lastZ = currentZ
            )
            return null
        }

        val deltaX = currentX - state.lastX
        val deltaZ = currentZ - state.lastZ
        val rawSpeed = sqrt(deltaX * deltaX + deltaZ * deltaZ).toFloat()
        val speedXZ = if (rawSpeed > 10f) 0f else rawSpeed

        val yawDeltaSigned = getAngleDeltaSigned(currentYaw, state.lastYaw)
        val pitchDeltaSigned = getAngleDeltaSigned(currentPitch, state.lastPitch)
        val yawDeltaAbs = abs(yawDeltaSigned)
        val pitchDeltaAbs = abs(pitchDeltaSigned)

        val yawAccel = abs(yawDeltaAbs - state.lastYawDeltaAbs)
        val pitchAccel = abs(pitchDeltaAbs - state.lastPitchDeltaAbs)

        state.lastYaw = currentYaw
        state.lastPitch = currentPitch
        state.lastYawDeltaAbs = yawDeltaAbs
        state.lastPitchDeltaAbs = pitchDeltaAbs
        state.lastX = currentX
        state.lastZ = currentZ

        val snapshot = PlayerSnapshot(
            yawDeltaAbs = yawDeltaAbs,
            pitchDeltaAbs = pitchDeltaAbs,
            yawDeltaSigned = yawDeltaSigned,
            pitchDeltaSigned = pitchDeltaSigned,
            yawAccel = yawAccel,
            pitchAccel = pitchAccel,
            speedXZ = speedXZ
        )

        val history = historySnapshot.computeIfAbsent(uuid) { ArrayDeque() }
        if (history.size >= maxSize) {
            history.removeFirst()
        }
        history.addLast(snapshot)

        return snapshot
    }

    fun getSnapshot(player: Player): List<PlayerSnapshot> {
        return historySnapshot[player.uniqueId]?.toList() ?: emptyList()
    }


    fun getYawEntropyAbs(player: Player, binCount: Int = 10): Double? {
        val history = historySnapshot[player.uniqueId] ?: return null
        if (history.size < 2) return null
        return calculateEntropyAbs(history, binCount) { it.yawDeltaAbs }
    }

    fun getPitchEntropyAbs(player: Player, binCount: Int = 10): Double? {
        val history = historySnapshot[player.uniqueId] ?: return null
        if (history.size < 2) return null
        return calculateEntropyAbs(history, binCount) { it.pitchDeltaAbs }
    }

    fun getYawEntropySigned(player: Player, binCount: Int = 10): Double? {
        val history = historySnapshot[player.uniqueId] ?: return null
        if (history.size < 2) return null
        return calculateEntropySigned(history, binCount) { it.yawDeltaSigned }
    }

    fun getPitchEntropySigned(player: Player, binCount: Int = 10): Double? {
        val history = historySnapshot[player.uniqueId] ?: return null
        if (history.size < 2) return null
        return calculateEntropySigned(history, binCount) { it.pitchDeltaSigned }
    }

    private inline fun calculateEntropyAbs(
        history: ArrayDeque<PlayerSnapshot>,
        binCount: Int,
        selector: (PlayerSnapshot) -> Float
    ): Double {
        if (binCount <= 0) return 0.0
        val bins = IntArray(binCount)
        val binWidth = 180.0 / binCount
        var total = 0

        for (snapshot in history) {
            val value = selector(snapshot).toDouble()
            val binIndex = if (value.isNaN()) {
                0
            } else {
                (value / binWidth).toInt().coerceIn(0, binCount - 1)
            }
            bins[binIndex]++
            total++
        }
        return entropyFromBins(bins, total)
    }

    private inline fun calculateEntropySigned(
        history: ArrayDeque<PlayerSnapshot>,
        binCount: Int,
        selector: (PlayerSnapshot) -> Float
    ): Double {
        if (binCount <= 0) return 0.0
        val bins = IntArray(binCount)
        val binWidth = 360.0 / binCount
        var total = 0

        for (snapshot in history) {
            val value = selector(snapshot).toDouble()
            val shifted = value + 180.0
            val binIndex = if (value.isNaN()) {
                0
            } else {
                (shifted / binWidth).toInt().coerceIn(0, binCount - 1)
            }
            bins[binIndex]++
            total++
        }
        return entropyFromBins(bins, total)
    }

    private fun entropyFromBins(bins: IntArray, total: Int): Double {
        if (total == 0) return 0.0
        var entropy = 0.0
        for (count in bins) {
            if (count > 0) {
                val p = count.toDouble() / total
                entropy -= p * log2(p)
            }
        }
        return entropy
    }

    fun getYawJitterAbs(player: Player): Double? {
        val history = historySnapshot[player.uniqueId] ?: return null
        if (history.size < 2) return null
        return calculateStdDev(history) { it.yawDeltaAbs }
    }

    fun getPitchJitterAbs(player: Player): Double? {
        val history = historySnapshot[player.uniqueId] ?: return null
        if (history.size < 2) return null
        return calculateStdDev(history) { it.pitchDeltaAbs }
    }

    fun getYawJitterSigned(player: Player): Double? {
        val history = historySnapshot[player.uniqueId] ?: return null
        if (history.size < 2) return null
        return calculateStdDev(history) { it.yawDeltaSigned }
    }

    fun getPitchJitterSigned(player: Player): Double? {
        val history = historySnapshot[player.uniqueId] ?: return null
        if (history.size < 2) return null
        return calculateStdDev(history) { it.pitchDeltaSigned }
    }

    private inline fun calculateStdDev(
        history: ArrayDeque<PlayerSnapshot>,
        selector: (PlayerSnapshot) -> Float
    ): Double? {
        val values = mutableListOf<Double>()
        var sum = 0.0
        for (snapshot in history) {
            val v = selector(snapshot).toDouble()
            if (!v.isFinite()) continue
            values.add(v)
            sum += v
        }
        val n = values.size
        if (n < 2) return null
        val mean = sum / n
        var varianceSum = 0.0
        for (v in values) {
            val diff = v - mean
            varianceSum += diff * diff
        }
        return sqrt(varianceSum / n)
    }

    fun getRotationSmoothnessYaw(player: Player): Double? {
        val history = historySnapshot[player.uniqueId] ?: return null
        if (history.size < 2) return null
        return calculateMaxAvgRatio(history) { it.yawDeltaAbs }
    }

    fun getRotationSmoothnessPitch(player: Player): Double? {
        val history = historySnapshot[player.uniqueId] ?: return null
        if (history.size < 2) return null
        return calculateMaxAvgRatio(history) { it.pitchDeltaAbs }
    }

    private inline fun calculateMaxAvgRatio(
        history: ArrayDeque<PlayerSnapshot>,
        selector: (PlayerSnapshot) -> Float
    ): Double? {
        var sum = 0.0
        var max = 0.0
        var count = 0
        for (snapshot in history) {
            val v = selector(snapshot).toDouble()
            if (!v.isFinite()) continue
            sum += v
            if (v > max) max = v
            count++
        }
        if (count == 0) return null
        val avg = sum / count
        if (avg == 0.0) return null
        return max / avg
    }

    fun getMicroAdjustCountYaw(player: Player): Int? {
        val history = historySnapshot[player.uniqueId] ?: return null
        var count = 0
        for (snapshot in history) {
            val delta = snapshot.yawDeltaAbs.toDouble()
            if (delta > 0.001 && delta < 0.5) count++
        }
        return count
    }

    fun getMicroAdjustCountPitch(player: Player): Int? {
        val history = historySnapshot[player.uniqueId] ?: return null
        var count = 0
        for (snapshot in history) {
            val delta = snapshot.pitchDeltaAbs.toDouble()
            if (delta > 0.001 && delta < 0.5) count++
        }
        return count
    }

    fun getJerkValue(player: Player): Double? {
        val history = historySnapshot[player.uniqueId] ?: return null
        val size = history.size
        if (size < 3) return null // нужно минимум 3 снимка для двух переходов accel

        var sumJerk = 0.0
        var count = 0
        // Используем итератор для эффективного обхода
        val iterator = history.iterator()
        var prev = iterator.next()
        while (iterator.hasNext()) {
            val current = iterator.next()
            val yawJerk = abs(current.yawAccel - prev.yawAccel)
            val pitchJerk = abs(current.pitchAccel - prev.pitchAccel)
            // Усреднение по осям для каждого перехода
            sumJerk += (yawJerk + pitchJerk) / 2.0
            count++
            prev = current
        }
        return if (count > 0) sumJerk / count else null
    }

    /**
     * Вычисляет R² линейной регрессии кумулятивной суммы yawDeltaSigned от индекса снимка.
     * Значение близкое к 1.0 означает, что поворот происходит с постоянной скоростью (читерский паттерн).
     * Для человека R² обычно значительно ниже.
     */
    fun getStraightLineRatio(player: Player): Double? {
        val history = historySnapshot[player.uniqueId] ?: return null
        val size = history.size
        if (size < 3) return null // нужно минимум 3 точки для осмысленной регрессии

        // Заполняем массив накопленных углов
        val yValues = DoubleArray(size)
        var cum = 0.0
        var index = 0
        for (snapshot in history) {
            cum += snapshot.yawDeltaSigned.toDouble()
            yValues[index++] = cum
        }

        // Первый проход: вычисляем суммы для коэффициентов регрессии
        var sumX = 0.0
        var sumY = 0.0
        var sumXY = 0.0
        var sumX2 = 0.0
        for (i in 0 until size) {
            val x = i.toDouble()
            val y = yValues[i]
            sumX += x
            sumY += y
            sumXY += x * y
            sumX2 += x * x
        }

        val n = size.toDouble()
        val meanX = sumX / n
        val meanY = sumY / n

        // Вычисляем общую сумму квадратов (SS_tot)
        var ssTot = 0.0
        for (i in 0 until size) {
            val diff = yValues[i] - meanY
            ssTot += diff * diff
        }
        if (ssTot == 0.0) return null // все значения одинаковы -> R² не определён

        // Вычисляем коэффициент наклона b и свободный член a
        val denominator = n * sumX2 - sumX * sumX
        if (denominator == 0.0) return null // теоретически не должно случиться, т.к. x разные
        val b = (n * sumXY - sumX * sumY) / denominator
        val a = meanY - b * meanX

        // Второй проход: вычисляем остаточную сумму квадратов (SS_res)
        var ssRes = 0.0
        for (i in 0 until size) {
            val x = i.toDouble()
            val yPred = a + b * x
            val diff = yValues[i] - yPred
            ssRes += diff * diff
        }

        val r2 = 1.0 - (ssRes / ssTot)
        // Защита от численных погрешностей – ограничиваем диапазон [0, 1]
        return r2.coerceIn(0.0, 1.0)
    }


    fun removePlayer(player: Player) {
        playerStates.remove(player.uniqueId)
        historySnapshot.remove(player.uniqueId)
        lastAimAngles.remove(player.uniqueId)
    }

    fun getSpeedXZ(player: Player): Float? {
        return historySnapshot[player.uniqueId]?.lastOrNull()?.speedXZ
    }
}