package zov.grusha.daynAC.ml

import zov.grusha.daynAC.tracking.HitData
import java.io.File

class DatasetManager(
    private val dataFolder: File,
    private val windowSize: Int = 16
) {

    private val legitFile = File(dataFolder, "legit.csv")
    private val cheatFile = File(dataFolder, "cheat.csv")
    private val legitVectorFile = File(dataFolder, "legit_vectors.csv")
    private val cheatVectorFile = File(dataFolder, "cheat_vectors.csv")

    init {
        if (!dataFolder.exists()) dataFolder.mkdirs()
        if (!legitFile.exists()) legitFile.writeText(header() + "\n")
        if (!cheatFile.exists()) cheatFile.writeText(header() + "\n")
        if (!legitVectorFile.exists()) legitVectorFile.writeText(vectorHeader() + "\n")
        if (!cheatVectorFile.exists()) cheatVectorFile.writeText(vectorHeader() + "\n")
    }

    private fun vectorHeader(): String = FeatureExtractor.csvHeader(windowSize)

    private fun header(): String {
        return listOf(
            "timestamp", "aimAngle", "distance", "hitTimeDelta", "hitTimeCV",
            "yawEntropyAbs", "pitchEntropyAbs", "yawEntropySigned", "pitchEntropySigned",
            "yawJitterAbs", "pitchJitterAbs", "yawJitterSigned", "pitchJitterSigned",
            "speedXZ", "rotSmoothYaw", "rotSmoothPitch", "snapFactor",
            "microAdjustYaw", "microAdjustPitch", "jerkValue", "straightLineRatio"
        ).joinToString(",")
    }

    fun writeSample(label: String, hit: HitData) {
        val targetFile = when (label) {
            "legit" -> legitFile
            "cheat" -> cheatFile
            else -> return // неизвестная метка — не пишем никуда
        }

        val row = listOf(
            hit.timestamp,
            hit.aimAngle,
            hit.distance,
            hit.hitTimeDelta,
            hit.hitTimeCV,
            hit.yawEntropyAbs,
            hit.pitchEntropyAbs,
            hit.yawEntropySigned,
            hit.pitchEntropySigned,
            hit.yawJitterAbs,
            hit.pitchJitterAbs,
            hit.yawJitterSigned,
            hit.pitchJitterSigned,
            hit.speedXZ,
            hit.rotSmoothYaw,
            hit.rotSmoothPitch,
            hit.snapFactor,
            hit.microAdjustYaw,
            hit.microAdjustPitch,
            hit.jerkValue,
            hit.straightLineRatio
        ).joinToString(",") { it?.toString() ?: "" }

        targetFile.appendText(row + "\n")
    }

    /**
     * Пишет вектор признаков окна ударов (windowSize * 20 + 7 значений).
     * Вызывается только когда [FeatureExtractor.extract] вернул непустой вектор —
     * строк в векторных CSV меньше, чем в поштучных, это нормально.
     */
    fun writeVectorSample(label: String, vector: FeatureExtractor.FeatureVector) {
        val targetFile = when (label) {
            "legit" -> legitVectorFile
            "cheat" -> cheatVectorFile
            else -> return // неизвестная метка — не пишем никуда
        }

        targetFile.appendText(vector.toCsvRow() + "\n")
    }

    /**
     * Читает все векторные сэмплы (legit_vectors.csv + cheat_vectors.csv)
     * для обучения.
     *
     * @return Triple(векторы, метки, счётчики реально распарсенных сэмплов
     *         по классам: (legit, cheat)). Битые строки (не та длина,
     *         нечисловые значения) пропускаются молча — поэтому счётчик
     *         строк файла и счётчик сэмплов могут расходиться; валидировать
     *         обучаемость датасета нужно по этим счётчикам, а не по строкам.
     */
    fun readVectorSamples(): Triple<List<DoubleArray>, List<Double>, Pair<Int, Int>> {
        val features = mutableListOf<DoubleArray>()
        val labels = mutableListOf<Double>()
        var legitParsed = 0
        var cheatParsed = 0

        fun loadFile(file: File, label: Double) {
            if (!file.exists()) return
            file.forEachLine { line ->
                val trimmed = line.trim()
                if (trimmed.isEmpty() || trimmed.startsWith("h0")) return@forEachLine // заголовок
                val parts = trimmed.split(',')
                if (parts.size != windowSize * FeatureExtractor.PER_HIT_FEATURES + FeatureExtractor.AGGREGATE_FEATURES) {
                    return@forEachLine
                }
                val vector = DoubleArray(parts.size) { i ->
                    val v = parts[i].toDoubleOrNull() ?: return@forEachLine
                    v
                }
                features.add(vector)
                labels.add(label)
                if (label == 0.0) legitParsed++ else cheatParsed++
            }
        }

        loadFile(legitVectorFile, 0.0)
        loadFile(cheatVectorFile, 1.0)

        return Triple(features, labels, Pair(legitParsed, cheatParsed))
    }

    /** Краткая статистика датасета для команды info. */
    fun getVectorDatasetStats(): Pair<Int, Int> {
        fun countLines(file: File): Int =
            if (!file.exists()) 0
            else file.readLines().count { it.isNotBlank() && !it.startsWith("h0") }

        return Pair(countLines(legitVectorFile), countLines(cheatVectorFile))
    }
}