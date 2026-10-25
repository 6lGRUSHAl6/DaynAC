package zov.grusha.daynAC.gui

import org.bukkit.Bukkit
import org.bukkit.GameMode
import org.bukkit.Location
import org.bukkit.Material
import org.bukkit.NamespacedKey
import org.bukkit.entity.Player
import org.bukkit.event.EventHandler
import org.bukkit.event.EventPriority
import org.bukkit.event.Listener
import org.bukkit.event.inventory.InventoryAction
import org.bukkit.event.inventory.InventoryClickEvent
import org.bukkit.event.inventory.InventoryCloseEvent
import org.bukkit.event.player.PlayerGameModeChangeEvent
import org.bukkit.event.player.PlayerQuitEvent
import org.bukkit.inventory.Inventory
import org.bukkit.inventory.ItemStack
import org.bukkit.inventory.meta.CompassMeta
import org.bukkit.persistence.PersistentDataType
import org.bukkit.plugin.Plugin
import zov.grusha.daynAC.DaynAC
import zov.grusha.daynAC.detection.DetectionEngine
import zov.grusha.daynAC.permissions.Permissions
import zov.grusha.daynAC.tracking.HitTracker
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap

class AllInfoGUI(
    private val plugin: Plugin,
    private val engine: DetectionEngine,
    private val hitTracker: HitTracker
) : Listener {

    companion object {
        private const val TITLE = "§8DaynAC — наблюдаемые игроки"
        private const val GUI_SIZE = 54

        /** 5 рядов (45 слотов) под карточки; 6-й ряд — под навигацию. */
        private const val PLAYERS_PER_PAGE = 45

        private const val NAV_PREV_SLOT = 45
        private const val NAV_INFO_SLOT = 49
        private const val NAV_NEXT_SLOT = 53

        /** Иконка по величине скора */
        fun scoreIcon(score: Double): String = when {
            score < 0.4 -> "§a✔"
            score < 0.65 -> "§e●"
            score < 0.9 -> "§c▲"
            else -> "§4⬛"
        }
    }

    /** UUID подозреваемого на компасе-карточке. */
    private val suspectKey = NamespacedKey(plugin, "suspect")

    /** Метка навигационной кнопки: "prev"/"next". Отсутствует у неактивной. */
    private val navKey = NamespacedKey(plugin, "nav")

    /**
     * Админ → его открытый GUI. Инвентарь опознаётся по объекту, а не по заголовку:
     * `InventoryView.getTitle()` deprecated и от версии к версии ведёт себя по-разному;
     * если бы проверка сломалась, клик не отменялся бы и админ мог бы вынуть предметы
     * из меню.
     */
    private val openGuis = ConcurrentHashMap<UUID, Inventory>()

    /**
     * Админ → текущая страница. Живёт дольше GUI: переоткрытие возвращает на ту же
     * страницу (удобно, когда админ разбирает длинный список и случайно закрыл меню).
     * Чистится только при выходе игрока с сервера — тогда состояние неактуально.
     */
    private val openPages = ConcurrentHashMap<UUID, Int>()

    /** Куда и в каком режиме вернуть админа после наблюдения. */
    private data class SpectateReturnPoint(val location: Location, val gameMode: GameMode)

    /**
     * Админ → точка, из которой он начал наблюдение. Хранится, пока админ находится
     * в spectator: смена режима вручную означает, что наблюдение закончено, и точка
     * удаляется, чтобы `/daynac reset` не телепортировал игрока неожиданно позже.
     */
    private val spectateReturnPoints = ConcurrentHashMap<UUID, SpectateReturnPoint>()

    /** Собирает и открывает GUI для администратора. */
    fun open(admin: Player): Inventory {
        val inventory = Bukkit.createInventory(null, GUI_SIZE, TITLE)
        openGuis[admin.uniqueId] = inventory
        populate(admin, inventory, openPages[admin.uniqueId] ?: 0)
        admin.openInventory(inventory)
        return inventory
    }

    /**
     * Возвращает админа из режима наблюдения в точку, откуда он его начал:
     * восстанавливает локацию и прежний игровой режим.
     *
     * @return true, если админ действительно наблюдал и был возвращён
     */
    fun resetSpectate(admin: Player): Boolean {
        val returnPoint = spectateReturnPoints.remove(admin.uniqueId) ?: return false
        // Сначала телепорт, потом режим: иначе админ окажется в survival
        // в точке наблюдения ещё до перемещения назад.
        admin.teleport(returnPoint.location)
        admin.gameMode = returnPoint.gameMode
        return true
    }

    /**
     * Снимок всего, что нужно и для отрисовки, и для сортировки.
     * Собирается один раз за рендер — engine.getScoreHistory не дёргается
     * повторно из разных мест.
     */
    private data class Card(
        val uuid: UUID,
        val name: String,
        val online: Boolean,
        val history: List<Double>,
        val hitCount: Int,
        val windowFill: Int
    ) {
        val maxScore: Double get() = history.maxOrNull() ?: 0.0
    }

    /**
     * Все карточки в детерминированном порядке:
     *   1) онлайн выше оффлайн — админ работает с теми, кого видит;
     *   2) внутри — по убыванию максимального скора (самые подозрительные сверху);
     *   3) при равенстве — по имени.
     *
     * Tie-break обязателен: без него `HashMap`-порядок `getTrackedPlayerIds()`
     * давал бы «перемешанные» страницы при каждом переоткрытии — пагинация без
     * стабильной сортировки бессмысленна. `sortedWith` в Kotlin стабильна.
     */
    private fun collectCards(): List<Card> {
        val ids = engine.getTrackedPlayerIds() + engine.hitCounts.keys
        return ids.asSequence().map { uuid ->
            Card(
                uuid = uuid,
                name = Bukkit.getOfflinePlayer(uuid).name ?: uuid.toString().take(8),
                online = Bukkit.getPlayer(uuid) != null,
                history = engine.getScoreHistory(uuid),
                hitCount = engine.getHitCount(uuid),
                windowFill = hitTracker.getWindowFill(uuid)
            )
        }.sortedWith(
            compareByDescending<Card> { it.online }
                .thenByDescending { it.maxScore }
                .thenBy { it.name.lowercase() }
        ).toList()
    }

    /**
     * Рисует страницу [requestedPage] в [inventory]. Индекс зажимается в допустимый
     * диапазон (список мог укоротиться: игрока забанили, TTL вычистил записи) и
     * сохраняется в [openPages] — следующий клик «вперёд/назад» считается от
     * фактически показанной страницы, а не от запрошенной.
     */
    private fun populate(admin: Player, inventory: Inventory, requestedPage: Int) {
        inventory.clear()

        val cards = collectCards()
        if (cards.isEmpty()) {
            fillEmptyState(inventory)
            openPages[admin.uniqueId] = 0
            return
        }

        val pages = (cards.size + PLAYERS_PER_PAGE - 1) / PLAYERS_PER_PAGE
        val page = requestedPage.coerceIn(0, pages - 1)
        openPages[admin.uniqueId] = page

        val from = page * PLAYERS_PER_PAGE
        val to = minOf(from + PLAYERS_PER_PAGE, cards.size)
        for (i in from until to) {
            inventory.setItem(i - from, cardToItem(cards[i]))
        }

        // Кнопок нет на единственной странице: незачем занимать ряд под неработающие стрелки.
        if (pages > 1) renderNavigation(inventory, page, pages, cards.size)
    }

    private fun cardToItem(card: Card): ItemStack =
        if (card.history.isEmpty()) watchItem(card) else compassItem(card)

    /** Игрок наблюдаем, но предсказаний ещё нет — окно признаков не заполнено. */
    private fun watchItem(card: Card): ItemStack {
        val item = ItemStack(Material.CLOCK)
        val meta = item.itemMeta
        meta.setDisplayName("§e${card.name} " + onlineTag(card.online))

        val lore = mutableListOf("§7Ударов: §f${card.windowFill}§7/§f${DaynAC.WINDOW_SIZE}")
        if (card.windowFill < DaynAC.WINDOW_SIZE) {
            lore.add("§7Предсказаний ещё нет — окно")
            lore.add("§7признаков не заполнено (нужно ${DaynAC.WINDOW_SIZE} ударов).")
        } else {
            // Окно полное, а предсказаний нет: анализ этого игрока не ведётся вовсе —
            // он в режиме записи датасета, либо детекция выключена.
            lore.add("§7Предсказаний нет — анализ не ведётся:")
            lore.add("§7игрок в режиме записи или детекция выключена.")
        }
        lore.add("§7Пинг: ${pingValue(card.uuid)}")

        meta.lore = lore
        item.itemMeta = meta
        return item
    }

    /** Полноценная карточка с историей ударов и кликабельным телепортом. */
    private fun compassItem(card: Card): ItemStack {
        // Активные флаги: по ним считаются урон и наказание. После сброса по простою
        // они пусты, хотя журнал сохранён — карточка это покажет.
        val active = engine.getRecentPredictions(card.uuid)
        val flagsCleared = active.isEmpty()
        val average = active.average() // NaN при снятых флагах — тогда в лоре прочерк

        val item = ItemStack(Material.COMPASS)
        val meta = item.itemMeta as CompassMeta
        meta.setDisplayName("§c${card.name} " + onlineTag(card.online))
        meta.lore = buildLore(
            maxScore = card.maxScore,
            average = average,
            history = card.history,
            hitCount = card.hitCount,
            uuid = card.uuid,
            flagsCleared = flagsCleared,
            windowFill = card.windowFill
        )
        meta.persistentDataContainer.set(suspectKey, PersistentDataType.STRING, card.uuid.toString())
        item.itemMeta = meta
        return item
    }

    private fun onlineTag(online: Boolean): String =
        if (online) "§a● В сети" else "§c● Не в сети"

    private fun renderNavigation(inventory: Inventory, page: Int, pages: Int, total: Int) {
        val hasPrev = page > 0
        val hasNext = page < pages - 1

        inventory.setItem(NAV_PREV_SLOT, navItem(
            label = if (hasPrev) "§a← Предыдущая" else "§7← Предыдущая",
            nav = if (hasPrev) "prev" else null,
            lore = if (hasPrev) listOf("§7Страница $page из $pages")
            else listOf("§7Вы в начале списка")
        ))
        inventory.setItem(NAV_INFO_SLOT, navItem(
            label = "§fСтраница §e${page + 1}§f/§e$pages",
            nav = null,
            lore = listOf("§7Игроков в списке: §f$total")
        ))
        inventory.setItem(NAV_NEXT_SLOT, navItem(
            label = if (hasNext) "§aСледующая →" else "§7Следующая →",
            nav = if (hasNext) "next" else null,
            lore = if (hasNext) listOf("§7Страница ${page + 2} из $pages")
            else listOf("§7Вы в конце списка")
        ))
    }

    /**
     * Навигационная кнопка. [nav] = null означает «неактивна»: PDC-метка не пишется,
     * обработчик клика её игнорирует — визуально серая и без реакции. Именно поэтому
     * реакция идёт по метке, а не по слоту: кнопка неотличима от декоративной стрелки
     * для случайного предмета в том же слоте, если layout когда-нибудь поменяется.
     */
    private fun navItem(label: String, nav: String?, lore: List<String>): ItemStack {
        val item = ItemStack(Material.ARROW)
        val meta = item.itemMeta
        meta.setDisplayName(label)
        meta.lore = lore
        if (nav != null) meta.persistentDataContainer.set(navKey, PersistentDataType.STRING, nav)
        item.itemMeta = meta
        return item
    }

    private fun fillEmptyState(inventory: Inventory) {
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

    private fun pingValue(uuid: UUID): String {
        val ping = Bukkit.getPlayer(uuid)?.ping ?: return "§8— §7(не в сети)"
        return "${pingColor(ping)}$ping§7 мс"
    }

    private fun pingColor(ping: Int): String = when {
        ping < 100 -> "§a"
        ping < 200 -> "§e"
        else -> "§c"
    }

    private fun fmt(v: Double): String = String.format(java.util.Locale.US, "%.3f", v)


    @EventHandler
    fun onInventoryClick(event: InventoryClickEvent) {
        val player = event.whoClicked as? Player ?: return
        val gui = openGuis[player.uniqueId] ?: return
        if (event.view.topInventory !== gui) return

        // Отменяем только взаимодействия с нашим (верхним) инвентарём — иначе админ
        // не может пользоваться собственным инвентарём, пока GUI открыт.
        // MOVE_TO_OTHER_INVENTORY (shift-клик) перекладывает в верхний инвентарь —
        // тоже блокируем. COLLECT_TO_CURSOR (двойной клик) собирает и из верхнего.
        val clickTop = event.clickedInventory === gui
        if (!player.hasPermission(Permissions.VIEW)) {
            // Право могли отозвать, пока GUI был открыт: меню закрываем, предметы не отдаём
            event.isCancelled = true
            player.closeInventory()
            return
        }
        if (clickTop ||
            event.action == InventoryAction.MOVE_TO_OTHER_INVENTORY ||
            event.action == InventoryAction.COLLECT_TO_CURSOR
        ) {
            event.isCancelled = true
        }

        if (!clickTop) return // клики по своему инвентарю — не наши
        val item = event.currentItem ?: return
        val meta = item.itemMeta ?: return
        val pdc = meta.persistentDataContainer

        // Навигация. Реагируем на PDC-метку: неактивная кнопка её не несёт, и клик
        // по ней просто проваливается дальше (не совпадёт ни с compass, ни с suspectKey).
        pdc.get(navKey, PersistentDataType.STRING)?.let { nav ->
            val current = openPages[player.uniqueId] ?: 0
            val target = when (nav) {
                "prev" -> current - 1
                "next" -> current + 1
                else -> return
            }
            // Re-render in-place: без close/open GUI не мерцает и не теряет фокус,
            // populate сам зажмёт target в допустимый диапазон.
            populate(player, gui, target)
            return
        }

        if (item.type != Material.COMPASS) return
        val uuidString = pdc.get(suspectKey, PersistentDataType.STRING) ?: return
        if (!player.hasPermission(Permissions.SPECTATE)) {
            player.sendMessage("§c[DaynAC] Недостаточно прав (${Permissions.SPECTATE}).")
            return
        }
        val target = Bukkit.getPlayer(UUID.fromString(uuidString)) ?: run {
            player.sendMessage("§c[DaynAC] Игрок не в сети — телепорт невозможен.")
            return
        }

        // Телепорт в режиме наблюдателя. Точку возврата запоминаем только при первом
        // клике: если админ переключается между игроками прямо из spectator, /daynac reset
        // всё равно вернёт его туда, откуда он начал наблюдение.
        player.closeInventory()
        openGuis.remove(player.uniqueId)

        spectateReturnPoints.putIfAbsent(
            player.uniqueId,
            SpectateReturnPoint(player.location.clone(), player.gameMode)
        )
        player.gameMode = GameMode.SPECTATOR
        player.teleport(target.location)
        player.sendMessage("§7[DaynAC] Наблюдение за §c${target.name}§7. Вернуться обратно: /daynac reset")
    }

    /** Чистка при закрытии GUI. Страницу не трогаем — переоткрытие вернёт админа туда же. */
    @EventHandler
    fun onClose(event: InventoryCloseEvent) {
        val gui = openGuis[event.player.uniqueId]
        if (gui != null && gui === event.inventory) {
            openGuis.remove(event.player.uniqueId)
        }
    }

    /** Смена режима вручную завершает наблюдение — точка возврата больше не актуальна. */
    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    fun onGameModeChange(event: PlayerGameModeChangeEvent) {
        if (event.newGameMode != GameMode.SPECTATOR) {
            spectateReturnPoints.remove(event.player.uniqueId)
        }
    }

    /** Полная очистка при выходе админа — состояние страницы больше не нужно. */
    @EventHandler
    fun onQuit(event: PlayerQuitEvent) {
        openGuis.remove(event.player.uniqueId)
        openPages.remove(event.player.uniqueId)
        spectateReturnPoints.remove(event.player.uniqueId)
    }
}