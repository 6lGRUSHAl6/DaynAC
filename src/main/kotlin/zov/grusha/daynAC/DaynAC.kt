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
import zov.grusha.daynAC.tracking.SnapshotTracker
import java.util.Locale
import java.util.UUID
import zov.grusha.daynAC.tracking.HitTracker
import zov.grusha.daynAC.ml.DatasetManager

class DaynAC : JavaPlugin(), Listener {

    private val hitCounter = mutableMapOf<Pair<UUID, UUID>, Int>()
    private val snapshotTracker = SnapshotTracker(maxSize = 20)
    private val hitTracker = HitTracker(maxSize = 8)

    private val datasetManager = DatasetManager(dataFolder)

    // ИЗМЕНЕНИЕ: Теперь храним статус записи для каждого игрока отдельно (по UUID)
    private val recordingPlayers = mutableMapOf<UUID, String>()

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
        if (command.name.equals("daynac", ignoreCase = true)) {
            when {
                args.isNotEmpty() && args[0] == "ver" -> {
                    sender.sendMessage("daynac v${description.version}")
                }
                args.isNotEmpty() && args[0] == "record" -> {
                    // ИЗМЕНЕНИЕ: Проверяем, что передано достаточно аргументов
                    if (args.size < 3) {
                        sender.sendMessage("Использование: /daynac record <ник игрока> <legit|cheat|off>")
                        return true
                    }

                    val targetName = args[1]
                    val targetPlayer = Bukkit.getPlayerExact(targetName)

                    if (targetPlayer == null) {
                        sender.sendMessage("Игрок $targetName не найден или не в сети.")
                        return true
                    }

                    val mode = args[2].lowercase()
                    when (mode) {
                        "legit", "cheat" -> {
                            recordingPlayers[targetPlayer.uniqueId] = mode
                            sender.sendMessage("Запись датасета для ${targetPlayer.name} включена в режиме: $mode")
                        }
                        "off" -> {
                            recordingPlayers.remove(targetPlayer.uniqueId)
                            sender.sendMessage("Запись датасета для ${targetPlayer.name} выключена.")
                        }
                        else -> {
                            sender.sendMessage("Использование: /daynac record <ник игрока> <legit|cheat|off>")
                        }
                    }
                }
                else -> {
                    sender.sendMessage("Неизвестная команда. Доступно: /daynac <ver|record>")
                }
            }
            return true
        }
        return false
    }

    // НОВЫЙ МЕТОД: Обработка подсказок (Tab Completion)
    override fun onTabComplete(
        sender: CommandSender,
        command: Command,
        alias: String,
        args: Array<out String>
    ): MutableList<String>? {
        if (command.name.equals("daynac", ignoreCase = true)) {
            when (args.size) {
                1 -> { // Подсказка для первого аргумента (ver, record)
                    return listOf("ver", "record")
                        .filter { it.startsWith(args[0], ignoreCase = true) }
                        .toMutableList()
                }
                2 -> { // Подсказка для второго аргумента (ник игрока, если первый record)
                    if (args[0].equals("record", ignoreCase = true)) {
                        return Bukkit.getOnlinePlayers()
                            .map { it.name }
                            .filter { it.startsWith(args[1], ignoreCase = true) }
                            .toMutableList()
                    }
                }
                3 -> { // Подсказка для третьего аргумента (режимы, если первый record)
                    if (args[0].equals("record", ignoreCase = true)) {
                        return listOf("legit", "cheat", "off")
                            .filter { it.startsWith(args[2], ignoreCase = true) }
                            .toMutableList()
                    }
                }
            }
        }
        return null
    }

    @EventHandler
    fun onEntityDamagePlayer(event: EntityDamageByEntityEvent) {
        val attacker = event.damager as? Player ?: return
        val victim = event.entity as? Player ?: return

        val hitData = snapshotTracker.buildHitData(attacker, victim)
        hitTracker.addHit(attacker, hitData)

        // ИЗМЕНЕНИЕ: Проверяем, записываем ли мы именно этого атакующего
        recordingPlayers[attacker.uniqueId]?.let { label ->
            datasetManager.writeSample(label, hitData)
        }

        val pairKey = Pair(attacker.uniqueId, victim.uniqueId)
        val currentHits = hitCounter.compute(pairKey) { _, count -> (count ?: 0) + 1 }

        val message = "[LOG] ${attacker.name} hit ${victim.name} (count=$currentHits) $hitData"

        Bukkit.getOnlinePlayers().filter { it.isOp }.forEach { it.sendMessage(message) }
        logger.info(message)
    }

    @EventHandler
    fun onPlayerQuit(event: PlayerQuitEvent) {
        snapshotTracker.removePlayer(event.player)
        hitTracker.removePlayer(event.player)
        recordingPlayers.remove(event.player.uniqueId)
    }

    private fun formatDouble(value: Double?, format: String = "%.4f"): String {
        return value?.let { String.format(Locale.US, format, it) } ?: "N/A"
    }
}