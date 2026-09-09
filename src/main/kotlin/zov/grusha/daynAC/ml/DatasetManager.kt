package zov.grusha.daynAC.ml

import zov.grusha.daynAC.tracking.HitData
import java.io.File

class DatasetManager(private val dataFolder: File) {

    private val legitFile = File(dataFolder, "legit.csv")
    private val cheatFile = File(dataFolder, "cheat.csv")

    init {
        if (!dataFolder.exists()) dataFolder.mkdirs()
        if (!legitFile.exists()) legitFile.writeText(header() + "\n")
        if (!cheatFile.exists()) cheatFile.writeText(header() + "\n")
    }

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
}