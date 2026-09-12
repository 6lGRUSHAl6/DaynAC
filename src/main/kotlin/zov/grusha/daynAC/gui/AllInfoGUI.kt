package zov.grusha.daynAC.gui

import org.bukkit.Bukkit
import org.bukkit.ChatColor
import org.bukkit.Material
import org.bukkit.entity.Player
import org.bukkit.event.EventHandler
import org.bukkit.event.Listener
import org.bukkit.event.inventory.InventoryClickEvent
import org.bukkit.inventory.Inventory
import org.bukkit.inventory.ItemStack
import org.bukkit.inventory.meta.CompassMeta
import org.bukkit.persistence.PersistentDataType
import org.bukkit.plugin.Plugin
import zov.grusha.daynAC.detection.DetectionEngine
import java.util.UUID

/**
 * GUI со списком всех наблюдаемых игроков и их скоров детекции.
 * Раздел 13 daynac.md: сундук-инвентарь, каждый игрок — компас,
 * ЛКМ по компасу — телепорт в режиме наблюдателя.
 *
 * Компас в ведущую руку не даёт — игрок сразу видит, куда телепортируется.
 */
class AllInfoGUI(private val plugin: Plugin, private val engine: DetectionEngine) : Listener {

    companion object {
        private const val TITLE = "§8DaynAC — наблюдаемые игроки"
        private const val GUI_SIZE = 54 // 6 рядов, максимум 54 игрока на страницу

        /** Иконка по величине скора — как в daynac.md. */
        fun scoreIcon(score: Double): String = when {
            score < 0.4 -> "§a✔"
            score < 0.65 -> "§e●"
            score < 0.9 -> "§c▲"
            else -> "§4⬛"
        }
    }

    /** Ключ для хранения UUID подозреваемого в PersistentDataContainer компаса. */
    private val suspectKey = org.bukkit.NamespacedKey(plugin, "suspect")

    private val openGuis = java.util.concurrent.ConcurrentHashMap.newKeySet<UUID>()

    /** Собирает и открывает GUI для администратора. */
    fun open(admin: Player) {
        val inventory = Bukkit.createInventory(null, GUI_SIZE, TITLE)
        populate(inventory)
        openGuis.add(admin.uniqueId)
        admin.openInventory(inventory)
    }

    private fun populate(inventory: Inventory) {
        // Все, кто хоть раз ударил: с предсказаниями и без (пока меньше 16 ударов —
        // предсказаний нет, но игрок уже наблюдается и это видно в GUI)
        val trackedIds = engine.getTrackedPlayerIds() + engine.hitCounts.keys

        for (uuid in trackedIds) {
            val recent = engine.getRecentPredictions(uuid)
            val hitCount = engine.getHitCount(uuid)
            val name = Bukkit.getOfflinePlayer(uuid).name ?: uuid.toString().take(8)

            if (recent.isEmpty()) {
                // Удары есть, но окно ещё не набралось — показываем прогресс
                val watch = ItemStack(Material.CLOCK)
                val meta = watch.itemMeta
                meta.setDisplayName("§e$name " + (if (isOnline(uuid)) "§a● В сети" else "§c● Не в сети"))
                meta.lore = listOf(
                    "§7Ударов: §f$hitCount§7/§f16",
                    "§7Предсказаний ещё нет — окно",
                    "§7признаков не заполнено (нужно 16 ударов)."
                )
                watch.itemMeta = meta
                if (inventory.firstEmpty() == -1) break
                inventory.setItem(inventory.firstEmpty(), watch)
                continue
            }

            val maxScore = recent.max()
            val average = recent.average()

            val compass = ItemStack(Material.COMPASS)
            val meta = compass.itemMeta as CompassMeta

            meta.setDisplayName("§c$name " + (if (isOnline(uuid)) "§a● В сети" else "§c● Не в сети"))
            meta.lore = buildLore(maxScore, average, recent, hitCount)
            meta.persistentDataContainer.set(suspectKey, PersistentDataType.STRING, uuid.toString())
            compass.itemMeta = meta

            if (inventory.firstEmpty() == -1) break // GUI полон
            inventory.setItem(inventory.firstEmpty(), compass)
        }

        // GUI пуст — объясняем почему
        if (inventory.isEmpty) {
            val filler = ItemStack(Material.GRAY_STAINED_GLASS_PANE)
            val meta = filler.itemMeta
            meta.setDisplayName("§7Нет данных")
            meta.lore = listOf(
                "§7Ни один игрок ещё не ударил другого",
                "§7игрока с момента запуска сервера.",
                "§8• модель не обучена (/daynac train)?",
                "§8• не было PvP-ударов",
                "§8• все ударившие — в режиме записи"
            )
            filler.itemMeta = meta
            for (slot in 0 until GUI_SIZE) inventory.setItem(slot, filler)
        }
    }

    private fun buildLore(maxScore: Double, average: Double, recent: List<Double>, hitCount: Int): List<String> {
        val lore = mutableListOf<String>()
        lore.add("▲ Макс: ${scoreIcon(maxScore)} ${fmt(maxScore)}")
        lore.add("")
        lore.add("Последние удары:")
        // Показываем последние 10, новые первыми — как в daynac.md
        recent.takeLast(10).asReversed().forEachIndexed { i, score ->
            lore.add(" #${i + 1} ${scoreIcon(score)} ${fmt(score)}")
        }
        lore.add("")
        lore.add("Среднее: ${fmt(average)}")
        lore.add("Всего ударов: $hitCount")
        lore.add("Вердикт: ${verdict(average)}")
        lore.add("")
        lore.add("§7ЛКМ — телепорт в режиме наблюдателя")
        return lore
    }

    private fun verdict(average: Double): String = when {
        average >= engine.punishThreshold -> "§4Опасный читер"
        average >= engine.flagThreshold -> "§cПодозрительный"
        average >= 0.4 -> "§eТребует наблюдения"
        else -> "§aСкорее всего легит"
    }

    private fun isOnline(uuid: UUID): Boolean = Bukkit.getPlayer(uuid) != null

    private fun fmt(v: Double): String = String.format(java.util.Locale.US, "%.3f", v)

    @EventHandler
    fun onInventoryClick(event: InventoryClickEvent) {
        val player = event.whoClicked as? Player ?: return
        if (player.uniqueId !in openGuis) return
        if (event.view.title != TITLE) return

        event.isCancelled = true // клики по GUI только читают

        if (event.currentItem == null || event.currentItem!!.type != Material.COMPASS) return
        val meta = event.currentItem!!.itemMeta ?: return
        val uuidString = meta.persistentDataContainer.get(suspectKey, PersistentDataType.STRING) ?: return
        val target = Bukkit.getPlayer(UUID.fromString(uuidString)) ?: run {
            player.sendMessage("§c[DaynAC] Игрок не в сети — телепорт невозможен.")
            return
        }

        // Телепорт в режиме наблюдателя
        player.closeInventory()
        openGuis.remove(player.uniqueId)

        player.gameMode = org.bukkit.GameMode.SPECTATOR
        player.teleport(target.location)
        player.sendMessage("§7[DaynAC] Наблюдение за §c${target.name}§7. Выйти из наблюдателя: /gamemode survival")
    }

    /** Очистка при выходе админа с открытым GUI. */
    @EventHandler
    fun onQuit(event: org.bukkit.event.player.PlayerQuitEvent) {
        openGuis.remove(event.player.uniqueId)
    }
}
