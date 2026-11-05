package zov.grusha.daynAC.tracking

import org.bukkit.Location
import org.bukkit.entity.Player
import kotlin.math.acos
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt
import org.bukkit.util.Vector

object CombatMath {

    /**
     * Точка на теле жертвы, в которую измеряется угол прицеливания.
     */
    enum class AimTargetPoint {
        /** Центр глаз (head) жертвы. */
        EYES,
        /** Центр хитбокса (середина тела). */
        BODY_CENTER
    }

    /**
     * Вычисляет угол между направлением взгляда атакующего и вектором
     * от его глаз до выбранной точки на теле жертвы.
     *
     * Угол измеряется в градусах и лежит в диапазоне [0, 180].
     * Значение 0 означает, что атакующий смотрит точно на выбранную точку.
     *
     * @param attacker   атакующий игрок
     * @param victim     жертва
     * @param targetPoint какая точка тела жертвы используется для расчёта
     * @return угол в градусах или null, если векторы не могут быть вычислены
     *         (например, игроки находятся в одной точке)
     */
    fun getAimAngle(attacker: Player, victim: Player, targetPoint: AimTargetPoint = AimTargetPoint.BODY_CENTER): Double? {
        val attackerEye = attacker.eyeLocation
        val direction = attackerEye.direction // уже нормализованный вектор взгляда

        // Определяем целевую точку на жертве
        val targetLocation = when (targetPoint) {
            AimTargetPoint.EYES -> victim.eyeLocation
            AimTargetPoint.BODY_CENTER -> getBodyCenter(victim)
        }

        val toTarget = targetLocation.toVector().subtract(attackerEye.toVector())

        // Если расстояние пренебрежимо мало, угол не определён
        if (toTarget.lengthSquared() < 1e-8) {
            return null
        }

        toTarget.normalize()

        val dot = direction.dot(toTarget)
        // Защита от погрешностей вычислений, которые могут вывести dot за пределы [-1, 1]
        val clampedDot = dot.coerceIn(-1.0, 1.0)
        val angleRad = acos(clampedDot)
        return Math.toDegrees(angleRad)
    }

    /** Результат [measure]: три величины одного удара, посчитанные по одному набору данных. */
    class HitGeometry(
        /** Угол прицеливания в центр тела жертвы (градусы) или null, если игроки в одной точке. */
        val aimAngle: Double?,
        /** Расстояние от глаз атакующего до центра хитбокса жертвы. */
        val distance: Double,
        /** Расстояние от глаз атакующего до ближайшей точки хитбокса жертвы. */
        val reachDistance: Double
    )

    /**
     * Считает [getAimAngle] (BODY_CENTER), [getDistance] и [getReachDistance] за один проход:
     * `eyeLocation` и `boundingBox` запрашиваются по одному разу, а вектор взгляда и
     * промежуточные точки считаются на примитивах без Location/Vector. Результаты совпадают
     * с отдельными функциями; на горячем пути (каждый удар) используется эта.
     */
    fun measure(attacker: Player, victim: Player): HitGeometry {
        val eye = attacker.eyeLocation
        val ex = eye.x
        val ey = eye.y
        val ez = eye.z
        val box = victim.boundingBox

        val dx = box.centerX - ex
        val dy = box.centerY - ey
        val dz = box.centerZ - ez
        val lengthSq = dx * dx + dy * dy + dz * dz
        val length = sqrt(lengthSq) // нужен и для угла, и для distance — считаем один раз

        // Тот же вектор взгляда, что у Location.getDirection(): единичный, из yaw/pitch.
        val pitch = Math.toRadians(eye.pitch.toDouble())
        val yaw = Math.toRadians(eye.yaw.toDouble())
        val horizontal = cos(pitch)
        val lookX = -horizontal * sin(yaw)
        val lookY = -sin(pitch)
        val lookZ = horizontal * cos(yaw)

        val aimAngle = if (lengthSq < 1e-8) {
            null
        } else {
            val dot = (lookX * dx + lookY * dy + lookZ * dz) / length
            Math.toDegrees(acos(dot.coerceIn(-1.0, 1.0)))
        }

        val rx = ex - ex.coerceIn(box.minX, box.maxX)
        val ry = ey - ey.coerceIn(box.minY, box.maxY)
        val rz = ez - ez.coerceIn(box.minZ, box.maxZ)

        return HitGeometry(aimAngle, length, sqrt(rx * rx + ry * ry + rz * rz))
    }

    private fun getBodyCenter(player: Player): Location {
        val box = player.boundingBox
        val center = box.center
        return Location(player.world, center.x, center.y, center.z)
    }

    fun getDistance(
        attacker: Player,
        victim: Player
    ): Double {
        val attackerEye = attacker.eyeLocation
        val victimCenter = getBodyCenter(victim)
        return attackerEye.distance(victimCenter)
    }

    fun getReachDistance(attacker: Player, victim: Player): Double {
        val eye = attacker.eyeLocation.toVector()
        val box = victim.boundingBox
        val closest = Vector(
            eye.x.coerceIn(box.minX, box.maxX),
            eye.y.coerceIn(box.minY, box.maxY),
            eye.z.coerceIn(box.minZ, box.maxZ)
        )
        return eye.distance(closest)
    }

}