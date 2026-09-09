package zov.grusha.daynAC.tracking

import org.bukkit.Location
import org.bukkit.entity.Player
import kotlin.math.acos

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

}