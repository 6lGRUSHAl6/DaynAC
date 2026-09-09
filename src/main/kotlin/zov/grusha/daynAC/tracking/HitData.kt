package zov.grusha.daynAC.tracking

data class HitData(
    val timestamp: Long,
    val aimAngle: Double?,
    val distance: Double,
    val hitTimeDelta: Long?,
    val hitTimeCV: Double?,
    val yawEntropyAbs: Double?,
    val pitchEntropyAbs: Double?,
    val yawEntropySigned: Double?,
    val pitchEntropySigned: Double?,
    val yawJitterAbs: Double?,
    val pitchJitterAbs: Double?,
    val yawJitterSigned: Double?,
    val pitchJitterSigned: Double?,
    val speedXZ: Float?,
    val rotSmoothYaw: Double?,
    val rotSmoothPitch: Double?,
    val snapFactor: Double?,
    val microAdjustYaw: Int?,
    val microAdjustPitch: Int?,
    val jerkValue: Double?,
    val straightLineRatio: Double?
)