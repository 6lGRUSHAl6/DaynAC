package zov.grusha.daynAC.tracking

data class PlayerSnapshot(
    val yawDeltaAbs: Float,      // величина поворота камеры (без учёта направления)
    val pitchDeltaAbs: Float,
    val yawDeltaSigned: Float,   // направление поворота: + вправо, - влево
    val pitchDeltaSigned: Float,
    val yawAccel: Float,         // изменение величины поворота (на основе abs)
    val pitchAccel: Float,
    val speedXZ: Float
)