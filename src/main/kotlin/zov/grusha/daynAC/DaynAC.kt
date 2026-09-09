package zov.grusha.daynAC

import org.bukkit.Bukkit
import org.bukkit.command.Command
import org.bukkit.command.CommandSender
import org.bukkit.entity.Player
import org.bukkit.event.EventHandler
import org.bukkit.event.Listener
import org.bukkit.event.entity.EntityDamageByEntityEvent
import org.bukkit.event.player.PlayerQuitEvent
import org.bukkit.plugin.java.JavaPlugin
import zov.grusha.daynAC.tracking.CombatMath
import zov.grusha.daynAC.tracking.SnapshotTracker
import java.util.Locale
import java.util.UUID

class DaynAC : JavaPlugin(), Listener {

    private val hitCounter = mutableMapOf<Pair<UUID, UUID>, Int>()
    private val snapshotTracker = SnapshotTracker(maxSize = 20)

    override fun onEnable() {
        logger.info("Enabling plugin...")
        server.pluginManager.registerEvents(this, this)
        Bukkit.getScheduler().runTaskTimer(this, Runnable {
            for (player in Bukkit.getOnlinePlayers()) {
                snapshotTracker.recordSnapshot(player)
            }
        }, 0L, 1L)
    }

    override fun onDisable() {
        logger.info("Disabling plugin...")
    }

    override fun onCommand(sender: CommandSender, command: Command, label: String, args: Array<out String>): Boolean {
        if (command.name.equals("daynac", ignoreCase = true) && args.isNotEmpty() && args[0] == "ver") {
            sender.sendMessage("daynac v${description.version}")
            return true
        }
        return false
    }

    @EventHandler
    fun onEntityDamagePlayer(event: EntityDamageByEntityEvent) {
        val attacker = event.damager as? Player ?: return
        val victim = event.entity as? Player ?: return

        val aimAngle = CombatMath.getAimAngle(attacker, victim)
        val distance = CombatMath.getDistance(attacker, victim)

        val (hitTimeDelta, prevAimAngle) = snapshotTracker.registerHitWithAngle(attacker, aimAngle)

        val snapFactor = if (prevAimAngle != null && aimAngle != null) {
            val safeAngle = aimAngle.coerceAtLeast(0.1)
            val ratio = prevAimAngle / safeAngle
            if (ratio.isFinite() && ratio > 0) Math.log1p(ratio.coerceAtMost(100.0)) else null
        } else null

        // Все метрики
        val hitTimeCV = snapshotTracker.getHitTimeCV(attacker)
        val yawEntropyAbs = snapshotTracker.getYawEntropyAbs(attacker)
        val pitchEntropyAbs = snapshotTracker.getPitchEntropyAbs(attacker)
        val yawEntropySigned = snapshotTracker.getYawEntropySigned(attacker)
        val pitchEntropySigned = snapshotTracker.getPitchEntropySigned(attacker)
        val yawJitterAbs = snapshotTracker.getYawJitterAbs(attacker)
        val pitchJitterAbs = snapshotTracker.getPitchJitterAbs(attacker)
        val yawJitterSigned = snapshotTracker.getYawJitterSigned(attacker)
        val pitchJitterSigned = snapshotTracker.getPitchJitterSigned(attacker)

        val speedXZ = snapshotTracker.getSpeedXZ(attacker)
        val speedXZText = formatDouble(speedXZ?.toDouble(), "%.3f b/t")

        val rotSmoothYaw = snapshotTracker.getRotationSmoothnessYaw(attacker)
        val rotSmoothPitch = snapshotTracker.getRotationSmoothnessPitch(attacker)
        val rotSmoothYawText = formatDouble(rotSmoothYaw, "%.3f")
        val rotSmoothPitchText = formatDouble(rotSmoothPitch, "%.3f")

        val microAdjustYaw = snapshotTracker.getMicroAdjustCountYaw(attacker)
        val microAdjustPitch = snapshotTracker.getMicroAdjustCountPitch(attacker)
        val microAdjustYawText = microAdjustYaw?.toString() ?: "N/A"
        val microAdjustPitchText = microAdjustPitch?.toString() ?: "N/A"

        val jerkValue = snapshotTracker.getJerkValue(attacker)
        val jerkText = formatDouble(jerkValue, "%.3f")

        // NEW: straightLineRatio (по Yaw)
        val straightLineRatio = snapshotTracker.getStraightLineRatio(attacker)
        val straightLineRatioText = formatDouble(straightLineRatio, "%.3f")

        // Счётчик ударов
        val pairKey = Pair(attacker.uniqueId, victim.uniqueId)
        val currentHits = hitCounter.compute(pairKey) { _, count -> (count ?: 0) + 1 }

        // Форматирование
        val aimText = formatDouble(aimAngle, "%.2f°")
        val distanceText = formatDouble(distance, "%.2f")
        val deltaText = hitTimeDelta?.toString() ?: "N/A"
        val hitTimeCVText = formatDouble(hitTimeCV, "%.3f")
        val yawEntropyAbsText = formatDouble(yawEntropyAbs)
        val pitchEntropyAbsText = formatDouble(pitchEntropyAbs)
        val yawEntropySignedText = formatDouble(yawEntropySigned)
        val pitchEntropySignedText = formatDouble(pitchEntropySigned)
        val yawJitterAbsText = formatDouble(yawJitterAbs)
        val pitchJitterAbsText = formatDouble(pitchJitterAbs)
        val yawJitterSignedText = formatDouble(yawJitterSigned)
        val pitchJitterSignedText = formatDouble(pitchJitterSigned)
        val snapFactorText = formatDouble(snapFactor, "%.2f")

        // Консольный лог
        val plainMessage = buildString {
            append("[LOG] ")
            append(attacker.name)
            append(" hit ")
            append(victim.name)
            append(" (count=").append(currentHits).append(") ")
            append("aim=").append(aimText).append(", ")
            append("hitDelta=").append(deltaText).append("ms, ")
            append("hitCV=").append(hitTimeCVText).append(", ")
            append("yawEntropyAbs=").append(yawEntropyAbsText).append(", ")
            append("pitchEntropyAbs=").append(pitchEntropyAbsText).append(", ")
            append("yawEntropySigned=").append(yawEntropySignedText).append(", ")
            append("pitchEntropySigned=").append(pitchEntropySignedText).append(", ")
            append("yawJitterAbs=").append(yawJitterAbsText).append(", ")
            append("speedXZ=").append(speedXZText).append(", ")
            append("pitchJitterAbs=").append(pitchJitterAbsText).append(", ")
            append("yawJitterSigned=").append(yawJitterSignedText).append(", ")
            append("pitchJitterSigned=").append(pitchJitterSignedText).append(", ")
            append("rotSmoothYaw=").append(rotSmoothYawText).append(", ")
            append("rotSmoothPitch=").append(rotSmoothPitchText).append(", ")
            append("snapFactor=").append(snapFactorText).append(", ")
            append("microAdjustYaw=").append(microAdjustYawText).append(", ")
            append("microAdjustPitch=").append(microAdjustPitchText).append(", ")
            append("jerk=").append(jerkText).append(", ")
            append("straightLineRatio=").append(straightLineRatioText).append(", ")
            append("distance=").append(distanceText)
        }

        // Цветное сообщение для операторов
        val coloredMessage = "§c[LOG] §fИгрок §e${attacker.name} §4ударил игрока §e${victim.name} " +
                "§7(удар #$currentHits) §b[" +
                "aim=$aimText, " +
                "Δt=${deltaText}ms, " +
                "hitCV=$hitTimeCVText, " +
                "distance=$distanceText, " +
                "yawEntropyAbs=$yawEntropyAbsText, " +
                "pitchEntropyAbs=$pitchEntropyAbsText, " +
                "yawEntropySigned=$yawEntropySignedText, " +
                "pitchEntropySigned=$pitchEntropySignedText, " +
                "spd=$speedXZText, " +
                "yawJitterAbs=$yawJitterAbsText, " +
                "pitchJitterAbs=$pitchJitterAbsText, " +
                "yawJitterSigned=$yawJitterSignedText, " +
                "pitchJitterSigned=$pitchJitterSignedText, " +
                "rotSmoothYaw=$rotSmoothYawText, " +
                "rotSmoothPitch=$rotSmoothPitchText, " +
                "snapFactor=$snapFactorText, " +
                "microAdjY=$microAdjustYawText, " +
                "microAdjP=$microAdjustPitchText, " +
                "jerk=$jerkText, " +
                "straightLineRatio=$straightLineRatioText" +
                "]"

        Bukkit.getOnlinePlayers()
            .filter { it.isOp }
            .forEach { op -> op.sendMessage(coloredMessage) }

        logger.info(plainMessage)
    }

    @EventHandler
    fun onPlayerQuit(event: PlayerQuitEvent) {
        snapshotTracker.removePlayer(event.player)
    }

    private fun formatDouble(value: Double?, format: String = "%.4f"): String {
        return value?.let { String.format(Locale.US, format, it) } ?: "N/A"
    }
}