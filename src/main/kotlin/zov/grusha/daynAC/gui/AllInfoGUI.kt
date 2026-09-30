package zov.grusha.daynAC.gui

import org.bukkit.Bukkit
import org.bukkit.ChatColor
import org.bukkit.Material
import org.bukkit.entity.Player
import org.bukkit.event.EventHandler
import org.bukkit.event.Listener
import org.bukkit.event.inventory.InventoryClickEvent
import org.bukkit.event.inventory.InventoryCloseEvent
import org.bukkit.event.inventory.InventoryAction
import org.bukkit.inventory.Inventory
import org.bukkit.inventory.ItemStack
import org.bukkit.inventory.meta.CompassMeta
import org.bukkit.persistence.PersistentDataType
import org.bukkit.plugin.Plugin
import zov.grusha.daynAC.DaynAC
import zov.grusha.daynAC.detection.DetectionEngine
import zov.grusha.daynAC.tracking.HitTracker
import java.util.UUID

/**
 * GUI со списком всех наблюдаемых игроков и их скоров детекции.
 * Раздел 13 daynac.md: сундук-инвентарь, каждый игрок — компас,
 * ЛКМ по компасу — телепорт в режиме наблюдателя.
 *
 * Компас в ведущую руку не даёт — игрок сразу видит, куда телепортируется.
 */
class AllInfoGUI(
    private val plugin: Plugin,
    private val engine: DetectionEngine,
    private val hitTracker: HitTracker
) : Listener {

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
            // Журнал скоров, а не активное окно: после сброса по простою флаги
            // сняты, но «Последние удары» игрока остаются для разбора.
            val history = engine.getScoreHistory(uuid)
            val hitCount = engine.getHitCount(uuid)
            val name = Bukkit.getOfflinePlayer(uuid).name ?: uuid.toString().take(8)

            if (history.isEmpty()) {
                // Предсказаний у игрока ещё не было. Прогресс берём из самого окна,
                // а не из счётчика ударов: счётчик суммарный, и после сброса по
                // простою окно могло обнулиться — «5/16» вводило бы в заблуждение.
                val watch = ItemStack(Material.CLOCK)
                val meta = watch.itemMeta
                val windowFill = hitTracker.getWindowFill(uuid)
                meta.setDisplayName("§e$name " + (if (isOnline(uuid)) "§a● В сети" else "§c● Не в сети"))

                val lore = mutableListOf("§7Ударов: §f$windowFill§7/§f${DaynAC.WINDOW_SIZE}")
                if (windowFill < DaynAC.WINDOW_SIZE) {
                    lore.add("§7Предсказаний ещё нет — окно")
                    lore.add("§7признаков не заполнено (нужно ${DaynAC.WINDOW_SIZE} ударов).")
                } else {
                    // Окно полное, а предсказаний нет: анализ этого игрока не ведётся
                    // вовсе — он в режиме записи датасета, либо детекция выключена.
                    lore.add("§7Предсказаний нет — анализ не ведётся:")
                    lore.add("§7игрок в режиме записи или детекция выключена.")
                }
                lore.add("§7Пинг: ${pingValue(uuid)}")

                meta.lore = lore
                watch.itemMeta = meta
                if (inventory.firstEmpty() == -1) break
                inventory.setItem(inventory.firstEmpty(), watch)
                continue
            }

            // Активные флаги: по ним считаются урон и наказание. После сброса по
            // простою они пусты, хотя журнал ударов сохранён — карточка это покажет.
            val active = engine.getRecentPredictions(uuid)
            val flagsCleared = active.isEmpty()
            val average = active.average() // NaN при снятых флагах — тогда в лоре прочерк

            val compass = ItemStack(Material.COMPASS)
            val meta = compass.itemMeta as CompassMeta

            meta.setDisplayName("§c$name " + (if (isOnline(uuid)) "§a● В сети" else "§c● Не в сети"))
            meta.lore = buildLore(
                maxScore = history.max(),
                average = average,
                history = history,
                hitCount = hitCount,
                uuid = uuid,
                flagsCleared = flagsCleared,
                windowFill = hitTracker.getWindowFill(uuid)
            )
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

    private fun buildLore(
        maxScore: Double,
        average: Double,
        history: List<Double>,
        hitCount: Int,
        uuid: UUID,
        flagsCleared: Boolean,
        windowFill: Int
    ): List<String> {
        val lore = mutableListOf<String>()
        // Макс — по журналу за всё время: после сброса по простою он остаётся
        // исторической отметкой и помечается, чтобы не читался как текущий флаг.
        lore.add("▲ Макс: ${scoreIcon(maxScore)} ${fmt(maxScore)}" + if (flagsCleared) " §7(история)" else "")
        lore.add("")
        lore.add("Последние удары:")
        // Показываем последние 10, новые первыми — как в daynac.md
        history.takeLast(10).asReversed().forEachIndexed { i, score ->
            lore.add(" #${i + 1} ${scoreIcon(score)} ${fmt(score)}")
        }
        lore.add("")
        if (flagsCleared) {
            // Флаги снял сброс по простою: показываем прогресс нового окна и
            // объясняем, почему предсказаний нет, — иначе карточка выглядит
            // как «данных нет» без причины.
            lore.add("Ударов: §f$windowFill§7/§f${DaynAC.WINDOW_SIZE} §7(окно набирается заново)")
            lore.add("§7Предсказаний нет — после сброса флагов")
            lore.add("§7окно признаков ещё не заполнено.")
            lore.add("Среднее: §7—")
        } else {
            lore.add("Среднее: ${fmt(average)}")
        }
        lore.add("Всего ударов: $hitCount")
        lore.add("Вердикт: " + if (flagsCleared) "§aФлаги сняты после простоя" else verdict(average))
        lore.add("Пинг: ${pingValue(uuid)}")
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

    /**
     * Готовое к вставке в лор значение пинга. Возвращает уже окрашенную строку
     * («§a45»), поэтому вызывающие не думают о цвете.
     *
     * У оффлайн-игрока соединения нет и Bukkit пинг не отдаёт — показываем прочерк,
     * а не 0: ноль читался бы как «идеальный пинг» у игрока, которого нет на сервере.
     */
    private fun pingValue(uuid: UUID): String {
        val ping = Bukkit.getPlayer(uuid)?.ping ?: return "§8— §7(не в сети)"
        return "${pingColor(ping)}$ping§7 мс"
    }

    /** Цвет пинга: зелёный — норма, жёлтый — заметная задержка, красный — лаги. */
    private fun pingColor(ping: Int): String = when {
        ping < 100 -> "§a"
        ping < 200 -> "§e"
        else -> "§c"
    }

    private fun fmt(v: Double): String = String.format(java.util.Locale.US, "%.3f", v)

    @EventHandler
    fun onInventoryClick(event: InventoryClickEvent) {
        val player = event.whoClicked as? Player ?: return
        if (player.uniqueId !in openGuis) return
        if (event.view.title != TITLE) return

        // Отменяем только взаимодействия с нашим (верхним) инвентарём —
        // иначе админ не может пользоваться собственным инвентарём,
        // пока GUI открыт.
        // MOVE_TO_OTHER_INVENTORY: shift-клик из нижнего инвентаря
        // перекладывает предметы в верхний — тоже блокируем.
        // COLLECT_TO_CURSOR (двойной клик) собирает предметы и из верхнего.
        val clickTop = event.clickedInventory === event.view.topInventory
        if (clickTop ||
            event.action == InventoryAction.MOVE_TO_OTHER_INVENTORY ||
            event.action == InventoryAction.COLLECT_TO_CURSOR
        ) {
            event.isCancelled = true
        }

        if (!clickTop) return // клики по своему инвентарю — не наши
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

    /** Чистка при закрытии GUI (ESC/кнопка) — иначе UUID зависал в openGuis навсегда. */
    @EventHandler
    fun onClose(event: InventoryCloseEvent) {
        if (event.view.title == TITLE) {
            openGuis.remove(event.player.uniqueId)
        }
    }

    /** Очистка при выходе админа с открытым GUI. */
    @EventHandler
    fun onQuit(event: org.bukkit.event.player.PlayerQuitEvent) {
        openGuis.remove(event.player.uniqueId)
    }
}
