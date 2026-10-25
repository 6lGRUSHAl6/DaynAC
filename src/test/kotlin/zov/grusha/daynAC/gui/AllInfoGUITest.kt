@file:Suppress("DEPRECATION")

package zov.grusha.daynAC.gui

import org.bukkit.GameMode
import org.bukkit.Location
import org.bukkit.Material
import org.bukkit.NamespacedKey
import org.bukkit.entity.Player
import org.bukkit.event.inventory.ClickType
import org.bukkit.event.inventory.InventoryAction
import org.bukkit.event.inventory.InventoryClickEvent
import org.bukkit.event.inventory.InventoryCloseEvent
import org.bukkit.event.inventory.InventoryType
import org.bukkit.event.player.PlayerGameModeChangeEvent
import org.bukkit.event.player.PlayerQuitEvent
import org.bukkit.inventory.Inventory
import org.bukkit.inventory.ItemStack
import org.bukkit.persistence.PersistentDataType
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.BeforeEach
import org.mockito.kotlin.any
import org.mockito.kotlin.mock
import org.mockito.kotlin.whenever
import org.mockbukkit.mockbukkit.MockBukkit
import org.mockbukkit.mockbukkit.ServerMock
import org.mockbukkit.mockbukkit.entity.PlayerMock
import org.mockbukkit.mockbukkit.plugin.PluginMock
import zov.grusha.daynAC.detection.DetectionEngine
import zov.grusha.daynAC.permissions.Permissions
import zov.grusha.daynAC.tracking.HitTracker
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * AllInfoGUI на MockBukkit: реальные инвентари, ItemMeta, PDC и события кликов.
 * DetectionEngine и HitTracker — моки: GUI только читает их данные.
 */
class AllInfoGUITest {

    private lateinit var server: ServerMock
    private lateinit var plugin: PluginMock
    private lateinit var engine: DetectionEngine
    private lateinit var hitTracker: HitTracker
    private lateinit var gui: AllInfoGUI

    private val tracked = linkedSetOf<UUID>()
    private val histories = mutableMapOf<UUID, List<Double>>()
    private val activeScores = mutableMapOf<UUID, List<Double>>()
    private val hitCounts = mutableMapOf<UUID, Int>()
    private val windowFills = mutableMapOf<UUID, Int>()
    private val engineHitCounts = ConcurrentHashMap<UUID, Int>()

    private val suspectKey get() = NamespacedKey(plugin, "suspect")
    private val navKey get() = NamespacedKey(plugin, "nav")

    @BeforeEach
    fun setUp() {
        server = MockBukkit.mock()
        plugin = MockBukkit.createMockPlugin("DaynAC")

        engine = mock()
        hitTracker = mock()
        whenever(engine.getTrackedPlayerIds()).thenAnswer { tracked.toSet() }
        whenever(engine.hitCounts).thenAnswer { engineHitCounts }
        whenever(engine.getScoreHistory(any())).thenAnswer { invocation ->
            histories[invocation.getArgument<UUID>(0)] ?: emptyList<Double>()
        }
        whenever(engine.getRecentPredictions(any())).thenAnswer { invocation ->
            activeScores[invocation.getArgument<UUID>(0)] ?: emptyList<Double>()
        }
        whenever(engine.getHitCount(any())).thenAnswer { invocation ->
            hitCounts[invocation.getArgument<UUID>(0)] ?: 0
        }
        whenever(hitTracker.getWindowFill(any())).thenAnswer { invocation ->
            windowFills[invocation.getArgument<UUID>(0)] ?: 0
        }
        whenever(engine.flagThreshold).thenReturn(0.65)
        whenever(engine.punishThreshold).thenReturn(0.95)

        gui = AllInfoGUI(plugin, engine, hitTracker)
    }

    @AfterEach
    fun tearDown() {
        MockBukkit.unmock()
    }

    // --- helpers -----------------------------------------------------------

    private fun track(
        uuid: UUID,
        history: List<Double> = emptyList(),
        active: List<Double> = history,
        hits: Int = history.size,
        windowFill: Int = history.size
    ) {
        tracked += uuid
        histories[uuid] = history
        activeScores[uuid] = active
        hitCounts[uuid] = hits
        windowFills[uuid] = windowFill
    }

    private fun admin(vararg permissions: String): PlayerMock {
        val player = server.addPlayer("Admin")
        for (permission in permissions) {
            player.addAttachment(plugin, permission, true)
        }
        player.recalculatePermissions()
        return player
    }

    private fun loreOf(item: ItemStack): List<String> = item.itemMeta?.lore ?: emptyList()

    private fun suspectOf(item: ItemStack?): UUID? {
        val pdc = item?.itemMeta?.persistentDataContainer ?: return null
        return pdc.get(suspectKey, PersistentDataType.STRING)?.let(UUID::fromString)
    }

    private fun compassSlot(inventory: Inventory, uuid: UUID): Int {
        for (slot in 0 until 45) {
            if (suspectOf(inventory.getItem(slot)) == uuid) return slot
        }
        error("Компас для $uuid не найден")
    }

    private fun click(admin: PlayerMock, rawSlot: Int, action: InventoryAction = InventoryAction.PICKUP_ALL): InventoryClickEvent =
        InventoryClickEvent(
            admin.getOpenInventory(),
            InventoryType.SlotType.CONTAINER,
            rawSlot,
            ClickType.LEFT,
            action
        )

    // --- Иконки и пустое состояние ----------------------------------------

    @Test
    fun `score icon boundaries match documented ranges`() {
        assertEquals("§a✔", AllInfoGUI.scoreIcon(0.0))
        assertEquals("§a✔", AllInfoGUI.scoreIcon(0.3999))
        assertEquals("§e●", AllInfoGUI.scoreIcon(0.4))
        assertEquals("§e●", AllInfoGUI.scoreIcon(0.6499))
        assertEquals("§c▲", AllInfoGUI.scoreIcon(0.65))
        assertEquals("§c▲", AllInfoGUI.scoreIcon(0.8999))
        assertEquals("§4⬛", AllInfoGUI.scoreIcon(0.9))
        assertEquals("§4⬛", AllInfoGUI.scoreIcon(1.0))
    }

    @Test
    fun `empty list fills all slots with glass panes`() {
        val inventory = gui.open(admin(Permissions.VIEW))

        assertEquals(54, inventory.size)
        for (slot in 0 until 54) {
            val item = assertNotNull(inventory.getItem(slot), "слот $slot")
            assertEquals(Material.GRAY_STAINED_GLASS_PANE, item.type)
        }
        assertEquals("§7Нет данных", inventory.getItem(0)!!.itemMeta!!.displayName)
    }

    // --- Карточки -----------------------------------------------------------

    @Test
    fun `tracked player without predictions renders a watch clock`() {
        val id = UUID.randomUUID()
        track(id, history = emptyList(), active = emptyList(), hits = 0, windowFill = 3)

        val inventory = gui.open(admin(Permissions.VIEW))
        val item = inventory.getItem(0)!!

        assertEquals(Material.CLOCK, item.type)
        val displayName = item.itemMeta!!.displayName
        assertTrue(displayName.contains("Не в сети"))
        assertTrue(displayName.startsWith("§e"), "имя должно быть подставлено, displayName=$displayName")
        assertTrue(loreOf(item).any { it.contains("Ударов: §f3§7/§f16") })
        assertTrue(loreOf(item).any { it.contains("Предсказаний ещё нет") })
    }

    @Test
    fun `full window without predictions explains recording mode`() {
        val id = UUID.randomUUID()
        track(id, history = emptyList(), active = emptyList(), hits = 20, windowFill = 16)

        val item = gui.open(admin(Permissions.VIEW)).getItem(0)!!

        assertEquals(Material.CLOCK, item.type)
        assertTrue(loreOf(item).any { it.contains("анализ не ведётся") })
        assertTrue(loreOf(item).any { it.contains("режиме записи") })
    }

    @Test
    fun `compass card shows history, average, hit count and uuid`() {
        val player = server.addPlayer("Cheater")
        val id = player.uniqueId
        track(id, history = listOf(0.2, 0.8), hits = 42, windowFill = 16)

        val inventory = gui.open(admin(Permissions.VIEW))
        val item = inventory.getItem(0)!!

        assertEquals(Material.COMPASS, item.type)
        assertEquals(id, suspectOf(item))
        assertTrue(item.itemMeta!!.displayName.contains("Cheater"))
        assertTrue(item.itemMeta!!.displayName.contains("В сети"))

        val lore = loreOf(item)
        assertTrue(lore.any { it.contains("Макс: §c▲ 0.800") })
        // Новые удары первыми.
        assertTrue(lore.any { it.contains("#1 §c▲ 0.800") })
        assertTrue(lore.any { it.contains("#2 §a✔ 0.200") })
        assertTrue(lore.any { it.contains("Среднее: 0.500") })
        assertTrue(lore.any { it.contains("Всего ударов: 42") })
        assertTrue(lore.any { it.contains("Пинг: §a0§7 мс") })
    }

    @Test
    fun `verdict text follows average score`() {
        val dangerous = server.addPlayer("Dangerous")
        val suspicious = server.addPlayer("Suspicious")
        val watch = server.addPlayer("Watch")
        val legit = server.addPlayer("Legit")

        track(dangerous.uniqueId, listOf(0.96))
        track(suspicious.uniqueId, listOf(0.70))
        track(watch.uniqueId, listOf(0.50))
        track(legit.uniqueId, listOf(0.10))

        val inventory = gui.open(admin(Permissions.VIEW))
        assertTrue(loreOf(inventory.getItem(compassSlot(inventory, dangerous.uniqueId))!!)
            .any { it.contains("Опасный читер") })
        assertTrue(loreOf(inventory.getItem(compassSlot(inventory, suspicious.uniqueId))!!)
            .any { it.contains("Подозрительный") })
        assertTrue(loreOf(inventory.getItem(compassSlot(inventory, watch.uniqueId))!!)
            .any { it.contains("Требует наблюдения") })
        assertTrue(loreOf(inventory.getItem(compassSlot(inventory, legit.uniqueId))!!)
            .any { it.contains("Скорее всего легит") })
    }

    @Test
    fun `cleared flags keep history in the card`() {
        val id = UUID.randomUUID()
        track(id, history = listOf(0.8), active = emptyList(), hits = 30, windowFill = 2)

        val item = gui.open(admin(Permissions.VIEW)).getItem(0)!!
        val lore = loreOf(item)

        assertTrue(lore.any { it.contains("(история)") })
        assertTrue(lore.any { it.contains("Флаги сняты после простоя") })
        assertTrue(lore.any { it.contains("Предсказаний нет — после сброса флагов") })
        assertTrue(lore.any { it.contains("Среднее: §7—") })
        assertTrue(lore.any { it.contains("окно набирается заново") })
    }

    // --- Сортировка и страницы ---------------------------------------------

    @Test
    fun `players are sorted online first then by max score`() {
        val onlineLow = server.addPlayer("Alice")
        val onlineHigh = server.addPlayer("Bob")
        val offline = UUID.randomUUID()

        engineHitCounts[offline] = 5 // tracked через hitCounts, без истории
        track(onlineLow.uniqueId, listOf(0.1))
        track(onlineHigh.uniqueId, listOf(0.8))

        val inventory = gui.open(admin(Permissions.VIEW))

        assertEquals(onlineHigh.uniqueId, suspectOf(inventory.getItem(0)))
        assertEquals(onlineLow.uniqueId, suspectOf(inventory.getItem(1)))
        // Оффлайн-игрок без истории — CLOCK, но всё равно после онлайна.
        assertEquals(Material.CLOCK, inventory.getItem(2)!!.type)
    }

    @Test
    fun `equal max scores are ordered by name case-insensitively`() {
        val bob = server.addPlayer("bob")
        val alice = server.addPlayer("Alice")
        track(bob.uniqueId, listOf(0.5))
        track(alice.uniqueId, listOf(0.5))

        val inventory = gui.open(admin(Permissions.VIEW))

        assertEquals(alice.uniqueId, suspectOf(inventory.getItem(0)))
        assertEquals(bob.uniqueId, suspectOf(inventory.getItem(1)))
    }

    @Test
    fun `single page has no navigation row`() {
        track(UUID.randomUUID(), listOf(0.5))

        val inventory = gui.open(admin(Permissions.VIEW))

        assertNull(inventory.getItem(45))
        assertNull(inventory.getItem(49))
        assertNull(inventory.getItem(53))
    }

    @Test
    fun `pagination renders arrows and switches pages in place`() {
        // 46 карточек: у каждой свой скор 0.01..0.46, сортировка по убыванию.
        val ids = (1..46).map { UUID.randomUUID() }
        ids.forEachIndexed { index, id -> track(id, listOf((index + 1) / 100.0)) }
        val admin = admin(Permissions.VIEW)
        val inventory = gui.open(admin)

        assertEquals("§7← Предыдущая", inventory.getItem(45)!!.itemMeta!!.displayName)
        assertEquals("§fСтраница §e1§f/§e2", inventory.getItem(49)!!.itemMeta!!.displayName)
        assertEquals("§aСледующая →", inventory.getItem(53)!!.itemMeta!!.displayName)
        val firstPageTop = suspectOf(inventory.getItem(0))
        assertEquals(ids[45], firstPageTop, "первым идёт самый высокий скор")

        gui.onInventoryClick(click(admin, 53))

        assertEquals("§a← Предыдущая", inventory.getItem(45)!!.itemMeta!!.displayName)
        assertEquals("§fСтраница §e2§f/§e2", inventory.getItem(49)!!.itemMeta!!.displayName)
        assertEquals("§7Следующая →", inventory.getItem(53)!!.itemMeta!!.displayName)
        assertEquals(ids[0], suspectOf(inventory.getItem(0)), "на второй странице остался самый низкий скор")

        gui.onInventoryClick(click(admin, 45))

        assertEquals("§fСтраница §e1§f/§e2", inventory.getItem(49)!!.itemMeta!!.displayName)
        assertEquals(ids[45], suspectOf(inventory.getItem(0)))
    }

    @Test
    fun `reopening returns to the last viewed page`() {
        val ids = (1..46).map { UUID.randomUUID() }
        ids.forEachIndexed { index, id -> track(id, listOf((index + 1) / 100.0)) }
        val admin = admin(Permissions.VIEW)

        gui.open(admin)
        gui.onInventoryClick(click(admin, 53))
        val reopened = gui.open(admin)

        assertEquals("§fСтраница §e2§f/§e2", reopened.getItem(49)!!.itemMeta!!.displayName)
    }

    @Test
    fun `page index is clamped when the list shrinks`() {
        val ids = (1..46).map { UUID.randomUUID() }
        ids.forEachIndexed { index, id -> track(id, listOf((index + 1) / 100.0)) }
        val admin = admin(Permissions.VIEW)
        gui.open(admin)
        gui.onInventoryClick(click(admin, 53))

        // Все, кроме одного, перестали отслеживаться (бан/TTL).
        tracked.clear()
        tracked += ids[0]
        val reopened = gui.open(admin)

        assertNull(reopened.getItem(49), "одна страница — навигации нет")
        assertEquals(ids[0], suspectOf(reopened.getItem(0)))
    }

    // --- Клики --------------------------------------------------------------

    @Test
    fun `click on own inventory is not cancelled`() {
        track(UUID.randomUUID(), listOf(0.5))
        val admin = admin(Permissions.VIEW)
        gui.open(admin)

        val event = click(admin, 54) // нижний инвентарь админа

        gui.onInventoryClick(event)

        assertTrue(!event.isCancelled())
    }

    @Test
    fun `shift click into gui is cancelled`() {
        track(UUID.randomUUID(), listOf(0.5))
        val admin = admin(Permissions.VIEW)
        gui.open(admin)

        val event = click(admin, 54, InventoryAction.MOVE_TO_OTHER_INVENTORY)

        gui.onInventoryClick(event)

        assertTrue(event.isCancelled())
    }

    @Test
    fun `click without view permission closes inventory and cancels`() {
        track(UUID.randomUUID(), listOf(0.5))
        val admin = admin() // без daynac.view
        gui.open(admin)

        val event = click(admin, 0)

        gui.onInventoryClick(event)

        assertTrue(event.isCancelled())
    }

    @Test
    fun `compass click teleports admin to online suspect in spectator`() {
        val target = server.addPlayer("Cheater")
        track(target.uniqueId, listOf(0.9))
        val admin = admin(Permissions.VIEW, Permissions.SPECTATE)
        val inventory = gui.open(admin)
        val slot = compassSlot(inventory, target.uniqueId)

        gui.onInventoryClick(click(admin, slot))

        assertEquals(GameMode.SPECTATOR, admin.gameMode)
        assertEquals(target.world, admin.world)
        assertTrue(admin.location.distance(target.location) < 0.01)
        assertTrue(admin.nextMessage().orEmpty().contains("Наблюдение за"))
    }

    @Test
    fun `reset returns admin to start location and previous game mode`() {
        val target = server.addPlayer("Cheater")
        track(target.uniqueId, listOf(0.9))
        val admin = admin(Permissions.VIEW, Permissions.SPECTATE)
        val start = Location(admin.world, 12.5, 70.0, -3.5)
        admin.teleport(start)
        admin.gameMode = GameMode.CREATIVE
        val inventory = gui.open(admin)

        gui.onInventoryClick(click(admin, compassSlot(inventory, target.uniqueId)))

        assertEquals(GameMode.SPECTATOR, admin.gameMode)
        assertTrue(gui.resetSpectate(admin))
        assertEquals(GameMode.CREATIVE, admin.gameMode)
        assertTrue(admin.location.distance(start) < 0.01)
        assertFalse(gui.resetSpectate(admin), "повторный reset без наблюдения ничего не делает")
    }

    @Test
    fun `switching observed players keeps the original return point`() {
        val first = server.addPlayer("First")
        val second = server.addPlayer("Second")
        track(first.uniqueId, listOf(0.9))
        track(second.uniqueId, listOf(0.8))
        val admin = admin(Permissions.VIEW, Permissions.SPECTATE)
        val start = Location(admin.world, 4.0, 64.0, 4.0)
        admin.teleport(start)

        gui.onInventoryClick(click(admin, compassSlot(gui.open(admin), first.uniqueId)))
        gui.onInventoryClick(click(admin, compassSlot(gui.open(admin), second.uniqueId)))

        assertTrue(gui.resetSpectate(admin))
        assertTrue(admin.location.distance(start) < 0.01, "reset должен вернуть к началу первого наблюдения")
    }

    @Test
    fun `leaving spectator manually drops the return point`() {
        val target = server.addPlayer("Cheater")
        track(target.uniqueId, listOf(0.9))
        val admin = admin(Permissions.VIEW, Permissions.SPECTATE)
        val inventory = gui.open(admin)
        gui.onInventoryClick(click(admin, compassSlot(inventory, target.uniqueId)))

        gui.onGameModeChange(PlayerGameModeChangeEvent(admin, GameMode.SURVIVAL))

        assertFalse(gui.resetSpectate(admin), "после ручного выхода из spectator reset не должен телепортировать")
    }

    @Test
    fun `compass click on offline suspect reports it`() {
        val id = UUID.randomUUID()
        track(id, listOf(0.9))
        val admin = admin(Permissions.VIEW, Permissions.SPECTATE)
        val inventory = gui.open(admin)
        val slot = compassSlot(inventory, id)

        gui.onInventoryClick(click(admin, slot))

        assertTrue(admin.nextMessage().orEmpty().contains("Игрок не в сети"))
        assertEquals(GameMode.SURVIVAL, admin.gameMode)
    }

    @Test
    fun `compass click without spectate permission reports missing permission`() {
        val target = server.addPlayer("Cheater")
        track(target.uniqueId, listOf(0.9))
        val admin = admin(Permissions.VIEW)
        val inventory = gui.open(admin)
        val slot = compassSlot(inventory, target.uniqueId)

        gui.onInventoryClick(click(admin, slot))

        assertTrue(admin.nextMessage().orEmpty().contains(Permissions.SPECTATE))
        assertEquals(GameMode.SURVIVAL, admin.gameMode)
    }

    // --- Закрытие и выход ----------------------------------------------------

    @Test
    fun `closing the gui forgets the open inventory`() {
        track(UUID.randomUUID(), listOf(0.5))
        val admin = admin(Permissions.VIEW)
        val inventory = gui.open(admin)
        val close: InventoryCloseEvent = mock()
        whenever(close.player).thenReturn(admin)
        whenever(close.inventory).thenReturn(inventory)

        gui.onClose(close)

        val event = click(admin, 0)
        gui.onInventoryClick(event)
        assertTrue(!event.isCancelled(), "после закрытия mock-событие клика не должно отменяться")
    }

    @Test
    fun `quit clears the remembered page`() {
        val ids = (1..46).map { UUID.randomUUID() }
        ids.forEachIndexed { index, id -> track(id, listOf((index + 1) / 100.0)) }
        val admin = admin(Permissions.VIEW)
        gui.open(admin)
        gui.onInventoryClick(click(admin, 53))

        val quit: PlayerQuitEvent = mock()
        whenever(quit.player).thenReturn(admin)
        gui.onQuit(quit)

        val reopened = gui.open(admin)
        // Страница сброшена на первую, карточки снова с самых высоких скоров.
        assertEquals("§fСтраница §e1§f/§e2", reopened.getItem(49)!!.itemMeta!!.displayName)
        assertEquals(ids[45], suspectOf(reopened.getItem(0)))
    }
}
