package zov.grusha.daynAC.detection

import org.bukkit.Bukkit
import org.bukkit.command.CommandSender
import org.bukkit.command.ConsoleCommandSender
import org.bukkit.configuration.file.YamlConfiguration
import org.bukkit.entity.Player
import org.bukkit.plugin.Plugin
import org.bukkit.plugin.java.JavaPlugin
import org.bukkit.scheduler.BukkitScheduler
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.BeforeEach
import com.destroystokyo.paper.profile.PlayerProfile
import org.bukkit.BanEntry
import org.mockito.ArgumentMatchers.contains
import org.mockito.MockedStatic
import org.mockito.Mockito
import org.mockito.kotlin.any
import org.mockito.kotlin.argumentCaptor
import org.mockito.kotlin.doAnswer
import org.mockito.kotlin.doThrow
import org.mockito.kotlin.eq
import org.mockito.kotlin.mock
import org.mockito.kotlin.never
import org.mockito.kotlin.times
import org.mockito.kotlin.verify
import org.mockito.kotlin.whenever
import zov.grusha.daynAC.ml.FeatureExtractor
import zov.grusha.daynAC.ml.NeuralNetwork
import zov.grusha.daynAC.permissions.Permissions
import java.util.UUID
import java.util.concurrent.AbstractExecutorService
import java.util.concurrent.CountDownLatch
import java.util.concurrent.ExecutorService
import java.util.concurrent.TimeUnit
import java.util.logging.Logger
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * DetectionEngine без Bukkit-сервера:
 *  - JavaPlugin и Player — Mockito-моки;
 *  - Bukkit статически замокан (scheduler, онлайн-игроки, getPlayer, консоль);
 *  - runTask выполняется inline — вердикт возвращается в тот же вызов, без ожидания тика;
 *  - время подменяется через инъекцию clock.
 */
class DetectionEngineTest {

    private lateinit var bukkit: MockedStatic<Bukkit>
    private lateinit var plugin: JavaPlugin
    private lateinit var network: NeuralNetwork
    private lateinit var scheduler: BukkitScheduler
    private lateinit var logger: Logger
    private lateinit var console: ConsoleCommandSender
    private var now = 1_000_000L
    private val onlinePlayers = mutableListOf<Player>()

    @BeforeEach
    fun setUp() {
        plugin = mock()
        logger = mock()
        whenever(plugin.logger).thenReturn(logger)
        whenever(plugin.isEnabled).thenReturn(true)

        network = mock()
        scheduler = mock()
        // Inference-пул зовёт runTask: выполняем задачу сразу, в текущем потоке.
        doAnswer { invocation ->
            invocation.getArgument<Runnable>(1).run()
            null
        }.whenever(scheduler).runTask(any<Plugin>(), any<Runnable>())

        bukkit = Mockito.mockStatic(Bukkit::class.java)
        bukkit.`when`<BukkitScheduler> { Bukkit.getScheduler() }.thenReturn(scheduler)
        bukkit.`when`<Collection<Player>> { Bukkit.getOnlinePlayers() }.thenAnswer { onlinePlayers }
        bukkit.`when`<Player?> { Bukkit.getPlayer(any<UUID>()) }.thenReturn(null)
        console = mock()
        bukkit.`when`<ConsoleCommandSender> { Bukkit.getConsoleSender() }.thenReturn(console)
    }

    @AfterEach
    fun tearDown() {
        bukkit.close()
    }

    // --- Конфиг и хелперы --------------------------------------------------

    /**
     * Кастомный конфиг с удобными числами: все пороги — точные двоичные дроби,
     * чтобы средние и множители урона совпадали с ожидаемыми без погрешности.
     */
    private fun custom(
        maxRecent: Int = 10,
        confirmWindows: Int = 2,
        punishEnabled: Boolean = true,
        action: String = "kick",
        banWithIp: Boolean = false,
        carryOver: Int = 60,
        inactivity: Int = 30,
        alertCooldown: Int = 5
    ): String = """
        detection:
          flag-threshold: 0.5
          reduce-threshold: 0.5
          cancel-threshold: 0.75
          punish-threshold: 0.75
          max-recent-predictions: $maxRecent
          punish-confirm-windows: $confirmWindows
          alert-cooldown-seconds: $alertCooldown
          inactivity-reset-seconds: $inactivity
          carry-over-seconds: $carryOver
          punish-enabled: $punishEnabled
          punish-action: $action
          ban-with-ip: $banWithIp
          punish-reason: test reason
    """.trimIndent()

    private fun createEngine(yaml: String = "", windowSize: Int = 4): DetectionEngine {
        val config = YamlConfiguration()
        if (yaml.isNotBlank()) config.loadFromString(yaml)
        whenever(plugin.config).thenReturn(config)
        return DetectionEngine(
            plugin,
            network,
            windowSize,
            clock = { now },
            inferencePool = DirectExecutorService()
        )
    }

    /**
     * Inference-пул, выполняющий задачи прямо в вызывающем потоке. Благодаря этому
     * вердикт и scheduler.runTask происходят синхронно, а статический мок Bukkit
     * (он thread-local) виден из analyzeHitAsync.
     */
    private class DirectExecutorService : AbstractExecutorService() {
        private var stopped = false

        override fun execute(command: Runnable) = command.run()
        override fun shutdown() { stopped = true }
        override fun shutdownNow(): MutableList<Runnable> {
            stopped = true
            return mutableListOf()
        }

        override fun isShutdown() = stopped
        override fun isTerminated() = stopped
        override fun awaitTermination(timeout: Long, unit: TimeUnit) = true
    }

    private fun player(name: String = "Tester"): Player {
        val p: Player = mock()
        whenever(p.uniqueId).thenReturn(UUID.randomUUID())
        whenever(p.name).thenReturn(name)
        whenever(p.isOnline).thenReturn(true)
        return p
    }

    private fun alertAdmin(name: String = "Admin"): Player {
        val p = player(name)
        whenever(p.hasPermission(Permissions.ALERTS)).thenReturn(true)
        return p
    }

    private fun vector(): FeatureExtractor.FeatureVector =
        FeatureExtractor.FeatureVector(DoubleArray(343), 16)

    /** Прогоняет вердикт через настоящий analyzeHitAsync и ждёт колбэк. */
    private fun verdicts(engine: DetectionEngine, player: Player, vararg probabilities: Double) {
        for (probability in probabilities) {
            whenever(network.predict(any<DoubleArray>())).thenReturn(probability)
            val latch = CountDownLatch(1)
            engine.analyzeHitAsync(player, vector()) { _, _ -> latch.countDown() }
            assertTrue(latch.await(2, TimeUnit.SECONDS), "inference не завершился за 2 с")
        }
    }

    // --- Валидация конфига -------------------------------------------------

    @Test
    fun `defaults are exposed when config is empty`() {
        val engine = createEngine()

        assertEquals(0.65, engine.flagThreshold)
        assertEquals(0.70, engine.reduceThreshold)
        assertEquals(0.90, engine.cancelThreshold)
        assertEquals(0.95, engine.punishThreshold)
        assertEquals(10, engine.maxRecentPredictions)
        assertEquals(2, engine.punishConfirmWindows)
        assertEquals(5, engine.alertCooldownSeconds)
        assertEquals(30, engine.inactivityResetSeconds)
        assertEquals(600, engine.carryOverSeconds)
        assertTrue(engine.punishEnabled)
        assertEquals("kick", engine.punishAction)
        assertFalse(engine.banWithIp)
    }

    @Test
    fun `punish action is normalized to lowercase`() {
        val engine = createEngine(custom(action = "KICK"))
        assertEquals("kick", engine.punishAction)
    }

    @Test
    fun `invalid punish action is rejected`() {
        assertFailsWith<IllegalArgumentException> { createEngine(custom(action = "explode")) }
    }

    @Test
    fun `thresholds must be inside zero-one range`() {
        assertFailsWith<IllegalArgumentException> {
            createEngine("detection:\n  flag-threshold: 1.5")
        }
    }

    @Test
    fun `flag must not exceed reduce`() {
        assertFailsWith<IllegalArgumentException> {
            createEngine("detection:\n  flag-threshold: 0.8")
        }
    }

    @Test
    fun `cancel must be strictly above reduce`() {
        assertFailsWith<IllegalArgumentException> {
            createEngine("detection:\n  reduce-threshold: 0.95")
        }
    }

    @Test
    fun `cancel must not exceed punish`() {
        assertFailsWith<IllegalArgumentException> {
            createEngine("detection:\n  cancel-threshold: 0.96")
        }
    }

    @Test
    fun `window size must be positive`() {
        assertFailsWith<IllegalArgumentException> { createEngine(windowSize = 0) }
    }

    @Test
    fun `recent prediction limit must be positive`() {
        assertFailsWith<IllegalArgumentException> {
            createEngine("detection:\n  max-recent-predictions: 0")
        }
    }

    @Test
    fun `confirm window count must be at least one`() {
        assertFailsWith<IllegalArgumentException> {
            createEngine("detection:\n  punish-confirm-windows: 0")
        }
    }

    @Test
    fun `inactivity reset must not be negative`() {
        assertFailsWith<IllegalArgumentException> {
            createEngine("detection:\n  inactivity-reset-seconds: -1")
        }
    }

    @Test
    fun `carry over must not be negative`() {
        assertFailsWith<IllegalArgumentException> {
            createEngine("detection:\n  carry-over-seconds: -1")
        }
    }

    // --- Счётчик ударов и простой -----------------------------------------

    @Test
    fun `hit count starts at zero and grows`() {
        val engine = createEngine()
        val p = player()

        assertEquals(0, engine.getHitCount(p.uniqueId))
        engine.recordHit(p)
        engine.recordHit(p)
        assertEquals(2, engine.getHitCount(p.uniqueId))
    }

    @Test
    fun `idle requires a hit and strictly more than configured seconds`() {
        val engine = createEngine()
        val p = player()

        assertFalse(engine.isIdle(p.uniqueId), "без ударов простоев не бывает")

        engine.recordHit(p)
        now += 30_000
        assertFalse(engine.isIdle(p.uniqueId), "ровно 30 секунд — ещё не простой")

        now += 1
        assertTrue(engine.isIdle(p.uniqueId))

        engine.resetFlags(p.uniqueId)
        assertFalse(engine.isIdle(p.uniqueId), "после сброса счётчик времени очищен")
    }

    @Test
    fun `idle detection can be disabled`() {
        val engine = createEngine(custom(inactivity = 0))
        val p = player()
        engine.recordHit(p)

        now += 999_999
        assertFalse(engine.isIdle(p.uniqueId))
    }

    // --- Множитель урона ---------------------------------------------------

    @Test
    fun `damage multiplier is one without predictions`() {
        val engine = createEngine()
        assertEquals(1.0, engine.getDamageMultiplier(player()))
    }

    @Test
    fun `damage multiplier is one below reduce threshold`() {
        val engine = createEngine(custom())
        val p = player()
        verdicts(engine, p, 0.4)

        assertEquals(1.0, engine.getDamageMultiplier(p), 1e-9)
    }

    @Test
    fun `damage multiplier at reduce threshold starts at one over 1_2`() {
        val engine = createEngine(custom())
        val p = player()
        verdicts(engine, p, 0.5)

        assertEquals(1.0 / 1.2, engine.getDamageMultiplier(p), 1e-9)
    }

    @Test
    fun `damage multiplier interpolates towards cancel threshold`() {
        val engine = createEngine(custom())
        val p = player()
        // Среднее (0.5 + 0.75)/2 = 0.625 -> t = 0.5.
        verdicts(engine, p, 0.5, 0.75)

        assertEquals(1.0 / 1.35, engine.getDamageMultiplier(p), 1e-9)
    }

    @Test
    fun `damage multiplier is zero at cancel threshold`() {
        val engine = createEngine(custom())
        val p = player()
        verdicts(engine, p, 0.75)

        assertEquals(0.0, engine.getDamageMultiplier(p), 1e-9)
    }

    @Test
    fun `recent predictions are capped but history keeps growing`() {
        val engine = createEngine(custom(maxRecent = 2))
        val p = player()
        verdicts(engine, p, 0.1, 0.2, 0.3)

        assertEquals(listOf(0.2, 0.3), engine.getRecentPredictions(p.uniqueId))
        assertEquals(0.3, engine.getLastPrediction(p))
        assertEquals(0.25, engine.getAveragePrediction(p)!!, 1e-9)
        assertEquals(listOf(0.1, 0.2, 0.3), engine.getScoreHistory(p.uniqueId))
        assertEquals(0.3, engine.getMaxPrediction(p.uniqueId)!!, 1e-9)
        assertTrue(engine.getTrackedPlayerIds().contains(p.uniqueId))
    }

    @Test
    fun `score history is capped at one hundred verdicts`() {
        val engine = createEngine()
        val p = player()

        verdicts(engine, p, *DoubleArray(50) { 0.1 })
        verdicts(engine, p, *DoubleArray(60) { 0.9 })

        val history = engine.getScoreHistory(p.uniqueId)
        assertEquals(100, history.size)
        assertEquals(0.1, history.first(), 1e-9)
        assertEquals(0.9, history.last(), 1e-9)
        assertEquals(0.9, engine.getMaxPrediction(p.uniqueId)!!, 1e-9)
    }

    @Test
    fun `unknown player has no predictions`() {
        val engine = createEngine()
        val p = player()

        assertNull(engine.getAveragePrediction(p))
        assertNull(engine.getLastPrediction(p))
        assertTrue(engine.getRecentPredictions(p.uniqueId).isEmpty())
        assertTrue(engine.getScoreHistory(p.uniqueId).isEmpty())
        assertNull(engine.getMaxPrediction(p.uniqueId))
        assertTrue(engine.getTrackedPlayerIds().isEmpty())
    }

    // --- Оповещения ---------------------------------------------------------

    @Test
    fun `flag alert goes only to players with alerts permission`() {
        val engine = createEngine(custom())
        val admin = alertAdmin("Admin")
        val regular = player("Regular")
        onlinePlayers += admin
        onlinePlayers += regular

        engine.handlePrediction(player("Cheater"), 0.6)

        verify(admin).sendMessage(contains("необычный удар"))
        verify(regular, never()).sendMessage(any<String>())
        verify(logger).warning(contains("необычный удар"))
    }

    @Test
    fun `no alert below flag threshold`() {
        val engine = createEngine(custom())
        val admin = alertAdmin()
        onlinePlayers += admin

        engine.handlePrediction(player("Cheater"), 0.4)

        verify(admin, never()).sendMessage(any<String>())
    }

    @Test
    fun `alert contains score, average and confirmation progress`() {
        val engine = createEngine(custom(), windowSize = 2)
        val admin = alertAdmin()
        onlinePlayers += admin
        val p = player("Cheater")

        engine.handlePrediction(p, 0.8)

        verify(admin).sendMessage(contains("0.800"))
        verify(admin).sendMessage(contains("средний"))
        verify(admin).sendMessage(contains("окна для наказания"))
    }

    @Test
    fun `alert cooldown throttles repeated flagged hits`() {
        val engine = createEngine(custom(), windowSize = 4)
        val admin = alertAdmin()
        onlinePlayers += admin
        val p = player("Cheater")

        // 0.6 выше flag (0.5), но ниже punish (0.75): серия не растёт и байпас
        // кулдауна не затрагивается.
        repeat(10) { engine.handlePrediction(p, 0.6) }

        // Первый удар ≥ flag-threshold оповещает мгновенно; остальные 9 происходят
        // в тот же момент фейковых часов (внутри кулдауна 5 с) и не создают
        // ни сообщения админам, ни строки в консольном логе.
        verify(admin, times(1)).sendMessage(any<String>())
        verify(logger, times(1)).warning(any<String>())
    }

    @Test
    fun `alert cooldown aggregates suppressed hits in next message`() {
        val engine = createEngine(custom(), windowSize = 4)
        val admin = alertAdmin()
        onlinePlayers += admin
        val p = player("Cheater")

        engine.handlePrediction(p, 0.9) // мгновенное оповещение
        engine.handlePrediction(p, 0.81) // кулдаун: подавлен и запомнен
        engine.handlePrediction(p, 0.99) // кулдаун: подавлен, становится максимумом

        now = 1_000_000L + 6_000 // 6 с спустя — кулдаун (5 с) истёк
        engine.handlePrediction(p, 0.87) // новое оповещение с агрегатом

        val captor = argumentCaptor<String>()
        verify(admin, times(2)).sendMessage(captor.capture())
        // Без цветовых кодов — чтобы проверять текст, а не раскраску.
        val plain = captor.secondValue.replace(Regex("§[0-9a-f]"), "")
        assertFalse(captor.firstValue.contains("ещё"), "у первого оповещения агрегата нет")
        assertTrue(plain.contains("ещё 2"), "число подавленных кулдауном ударов")
        assertTrue(plain.contains("0.99"), "максимальный скор среди подавленных")
    }

    @Test
    fun `cooldown zero alerts on every flagged hit`() {
        val engine = createEngine(custom(alertCooldown = 0))
        val admin = alertAdmin()
        onlinePlayers += admin
        val p = player("Cheater")

        repeat(3) { engine.handlePrediction(p, 0.9) }

        // 0 = прежнее поведение: оповещение на каждый удар без кулдауна.
        verify(admin, times(3)).sendMessage(any<String>())
        verify(logger, times(3)).warning(any<String>())
    }

    @Test
    fun `alert cooldown must not be negative`() {
        assertFailsWith<IllegalArgumentException> {
            createEngine("detection:\n  alert-cooldown-seconds: -1")
        }
    }

    @Test
    fun `final punish confirmation bypasses alert cooldown`() {
        val engine = createEngine(custom(), windowSize = 2)
        val admin = alertAdmin()
        onlinePlayers += admin
        val p = player("Cheater")

        engine.handlePrediction(p, 0.6) // обычный подозрительный удар: алерт, кулдаун активен
        engine.handlePrediction(p, 0.8) // открытие серии (0→1) — под кулдауном, как обычное оповещение
        engine.handlePrediction(p, 0.6) // тоже под кулдауном: подавлен, запомнен как агрегат
        val punished = engine.handlePrediction(p, 0.8) // 1→2 — последнее окно перед наказанием

        assertTrue(punished, "ветка наказания сработала")
        val captor = argumentCaptor<String>()
        verify(admin, times(2)).sendMessage(captor.capture())
        val plain = captor.secondValue.replace(Regex("§[0-9a-f]"), "")
        assertTrue(plain.contains("2/2"), "статус окон 2/2 виден админам вовремя")
        assertTrue(plain.contains("наказание"), "статус «наказание» виден до кика")
        assertTrue(plain.contains("ещё 2"), "байпас выпустил накопленный агрегат")
    }

    @Test
    fun `observation mode keeps cooldown on punish-threshold hits`() {
        // punish-enabled: false: серия расходуется при подтверждении и открывается
        // заново каждые ~windowSize ударов. Рост без наказания не байпасится —
        // иначе наблюдательный режим возвращал бы спам.
        val engine = createEngine(custom(punishEnabled = false), windowSize = 2)
        val admin = alertAdmin()
        onlinePlayers += admin
        val p = player("Cheater")

        // 20 ударов = около десятка циклов open→confirm→open.
        repeat(20) { engine.handlePrediction(p, 0.9) }

        verify(admin, times(1)).sendMessage(any<String>())
        verify(logger, times(1)).warning(any<String>())
    }

    @Test
    fun `punish action none keeps cooldown on punish-threshold hits`() {
        // punish-action: none: серия циклится так же, как при punish-enabled: false.
        val engine = createEngine(custom(action = "none"), windowSize = 2)
        val admin = alertAdmin()
        onlinePlayers += admin
        val p = player("Cheater")

        repeat(20) { engine.handlePrediction(p, 0.9) }

        verify(admin, times(1)).sendMessage(any<String>())
        verify(logger, times(1)).warning(any<String>())
    }

    @Test
    fun `first alert is instant even with small clock origin`() {
        // lastAlertTimeMillis == 0 нельзя считать активным кулдауном: на часах,
        // отсчитываемых от малого значения (шов clock в тестах), проверка
        // now - 0 < cooldown подавила бы первое оповещение.
        now = 2_500
        val engine = createEngine(custom())
        val admin = alertAdmin()
        onlinePlayers += admin
        val p = player("Cheater")

        engine.handlePrediction(p, 0.6)

        verify(admin, times(1)).sendMessage(any<String>())
    }

    @Test
    fun `punishment sends chat notice to admins`() {
        val engine = createEngine()
        val admin = alertAdmin()
        onlinePlayers += admin
        val p = player("Cheater")

        engine.punishPlayer(p)

        verify(admin).sendMessage(contains("авто-кик"))
        // Уведомление не дублируется в лог: у ветки наказания своя консольная строка.
        verify(logger, times(1)).warning(any<String>())
    }

    @Test
    fun `idle reset flushes pending suppressed aggregate`() {
        val engine = createEngine(custom(), windowSize = 4)
        val admin = alertAdmin()
        onlinePlayers += admin
        val p = player("Cheater")
        bukkit.`when`<Player?> { Bukkit.getPlayer(p.uniqueId) }.thenReturn(p)

        engine.handlePrediction(p, 0.9)  // оповещение (рост серии 0→1 — тоже мимо кулдауна)
        engine.handlePrediction(p, 0.81) // подавлен кулдауном
        engine.resetFlags(p.uniqueId)    // бой окончен: итоговое сообщение с агрегатом

        now = 1_000_000L + 6_000 // новый бой, кулдаун снят целиком
        engine.handlePrediction(p, 0.9) // мгновенное оповещение без агрегата

        val captor = argumentCaptor<String>()
        verify(admin, times(3)).sendMessage(captor.capture())
        val flushed = captor.secondValue.replace(Regex("§[0-9a-f]"), "")
        assertTrue(flushed.contains("ещё 1"), "итоговое сообщение показывает подавленный удар")
        assertTrue(flushed.contains("0.81"), "максимум среди подавленных ударов")
        assertFalse(captor.thirdValue.contains("ещё"), "новый бой начинается с чистым листом оповещений")
    }

    // --- Серия независимых окон для наказания ------------------------------

    @Test
    fun `punishment needs two independent windows and consumes the series`() {
        val engine = createEngine(custom(), windowSize = 2)
        val p = player()

        assertFalse(engine.handlePrediction(p, 0.8), "первое окно только открывает серию")
        assertFalse(engine.handlePrediction(p, 0.8), "окна ещё пересекаются")
        assertTrue(engine.handlePrediction(p, 0.8), "второе независимое окно подтверждает наказание")
        assertFalse(engine.handlePrediction(p, 0.8), "серия израсходована, начинается заново")
    }

    @Test
    fun `clean window resets the series after grace period`() {
        val engine = createEngine(custom(), windowSize = 4)
        val p = player()

        assertFalse(engine.handlePrediction(p, 0.8)) // confirmed = 1
        // Вердикты ниже порога: серия рвётся только после windowSize + confirmGrace (4 + 2).
        repeat(6) { assertFalse(engine.handlePrediction(p, 0.1)) }
        // Если бы серия не сбросилась, этот вердикт дал бы confirmed = 2 и наказание.
        assertFalse(engine.handlePrediction(p, 0.8), "серия должна была сброситься")
    }

    @Test
    fun `punish disabled never returns true but still consumes the series`() {
        val engine = createEngine(custom(punishEnabled = false), windowSize = 2)
        val p = player()

        assertFalse(engine.handlePrediction(p, 0.8))
        assertFalse(engine.handlePrediction(p, 0.8))
        assertFalse(engine.handlePrediction(p, 0.8), "режим наблюдения не наказывает")
        assertFalse(engine.handlePrediction(p, 0.8), "серия всё равно израсходована")
    }

    // --- resetFlags и carry-over ------------------------------------------

    @Test
    fun `reset flags keeps history but clears active predictions`() {
        val engine = createEngine(custom())
        val p = player()
        verdicts(engine, p, 0.6)

        engine.resetFlags(p.uniqueId)

        assertNull(engine.getAveragePrediction(p))
        assertTrue(engine.getRecentPredictions(p.uniqueId).isEmpty())
        assertEquals(listOf(0.6), engine.getScoreHistory(p.uniqueId))
        assertTrue(engine.getTrackedPlayerIds().contains(p.uniqueId), "журнал переживает сброс")
    }

    @Test
    fun `carry over applies after idle reset and decays to nothing`() {
        val engine = createEngine(custom(carryOver = 60))
        val p = player()
        verdicts(engine, p, 0.5, 0.9)
        val average = engine.getAveragePrediction(p)!!
        assertTrue(average >= engine.reduceThreshold)

        engine.resetFlags(p.uniqueId)
        val expectedInitial = 1.0 / (1.2 + (average - 0.5) / 0.25 * 0.3)
        assertEquals(expectedInitial, engine.getDamageMultiplier(p), 1e-9, "сразу после сброса")

        // Четверть времени жизни: скор линейно затухает до 0.75 * average.
        now += 15_000
        val carried = average * 0.75
        val expectedQuarter = 1.0 / (1.2 + (carried - 0.5) / 0.25 * 0.3)
        assertEquals(expectedQuarter, engine.getDamageMultiplier(p), 1e-9)

        now += 46_000 // всего 61 с > 60 с
        assertEquals(1.0, engine.getDamageMultiplier(p), 1e-9)
    }

    @Test
    fun `clean window does not create carry over`() {
        val engine = createEngine(custom(carryOver = 60))
        val p = player()
        verdicts(engine, p, 0.4)

        engine.resetFlags(p.uniqueId)

        assertEquals(1.0, engine.getDamageMultiplier(p), 1e-9)
    }

    @Test
    fun `manual clear removes existing carry over`() {
        val engine = createEngine(custom(carryOver = 60))
        val p = player()
        verdicts(engine, p, 0.5, 0.9)
        engine.resetFlags(p.uniqueId) // idle-сброс: carry-over сохранён
        assertTrue(engine.getDamageMultiplier(p) < 1.0)

        engine.resetFlags(p.uniqueId, keepCarryOver = false)

        assertEquals(1.0, engine.getDamageMultiplier(p), 1e-9)
    }

    @Test
    fun `carry over is disabled by zero lifetime`() {
        val engine = createEngine(custom(carryOver = 0))
        val p = player()
        verdicts(engine, p, 0.5, 0.9)

        engine.resetFlags(p.uniqueId)

        assertEquals(1.0, engine.getDamageMultiplier(p), 1e-9)
    }

    // --- Асинхронный анализ -------------------------------------------------

    @Test
    fun `async verdict is recorded and passed to callback`() {
        val engine = createEngine()
        val p = player()
        whenever(network.predict(any<DoubleArray>())).thenReturn(0.42)

        var callbackProbability: Double? = null
        val latch = CountDownLatch(1)
        engine.analyzeHitAsync(p, vector()) { _, probability ->
            callbackProbability = probability
            latch.countDown()
        }

        assertTrue(latch.await(2, TimeUnit.SECONDS))
        assertEquals(0.42, callbackProbability)
        assertEquals(0.42, engine.getLastPrediction(p))
    }

    @Test
    fun `non-finite prediction is ignored`() {
        val engine = createEngine()
        val p = player()
        verdicts(engine, p, 0.3)

        whenever(network.predict(any<DoubleArray>())).thenReturn(Double.NaN)
        var called = false
        engine.analyzeHitAsync(p, vector()) { _, _ -> called = true }

        verify(network, times(2)).predict(any<DoubleArray>())
        assertFalse(called)
        assertEquals(0.3, engine.getLastPrediction(p))
        assertEquals(1, engine.getRecentPredictions(p.uniqueId).size)
    }

    @Test
    fun `inference exception is logged and ignored`() {
        val engine = createEngine()
        val p = player()
        verdicts(engine, p, 0.3)

        whenever(network.predict(any<DoubleArray>())).thenThrow(IllegalStateException("boom"))
        var called = false
        engine.analyzeHitAsync(p, vector()) { _, _ -> called = true }

        verify(logger).warning(contains("Inference failed"))
        assertFalse(called)
        assertEquals(0.3, engine.getLastPrediction(p))
    }

    @Test
    fun `verdict is dropped when plugin is disabled`() {
        val engine = createEngine()
        val p = player()
        whenever(plugin.isEnabled).thenReturn(false)
        whenever(network.predict(any<DoubleArray>())).thenReturn(0.9)

        var called = false
        engine.analyzeHitAsync(p, vector()) { _, _ -> called = true }

        verify(network).predict(any<DoubleArray>())
        assertFalse(called)
        assertNull(engine.getLastPrediction(p))
    }

    // --- Наказания ----------------------------------------------------------

    @Test
    fun `default kick message reproduces the previous hardcoded text`() {
        val engine = createEngine(custom()) // punish-reason: test reason
        val p = player("Victim")
        verdicts(engine, p, 0.75) // средний = 0.75 → {score} = 75.0%

        engine.punishPlayer(p)

        val captor = argumentCaptor<String>()
        verify(p).kickPlayer(captor.capture())
        assertEquals(
            "§c[DaynAC] §ftest reason\n" +
                "§7Средняя вероятность: §f75.0%\n" +
                "§7Если это ошибка — сообщите администрации.",
            captor.firstValue,
            "шаблон по умолчанию дублирует прежний собранный вручную текст",
        )
    }

    @Test
    fun `kick message uses punish-message template with placeholders`() {
        val engine = createEngine(
            """
            detection:
              punish-reason: test reason
              punish-message:
                kick: "&c{player} was {action}-ed: {reason} @{score}"
            """.trimIndent()
        )
        val p = player("Victim")
        verdicts(engine, p, 0.75)

        engine.punishPlayer(p)

        val captor = argumentCaptor<String>()
        verify(p).kickPlayer(captor.capture())
        assertEquals("§cVictim was kick-ed: test reason @75.0%", captor.firstValue)
    }

    @Test
    fun `ban message uses punish-message list with one line per element`() {
        val engine = createEngine(
            """
            detection:
              punish-action: ban
              punish-reason: test reason
              punish-message:
                ban:
                  - "&6Line one {score}"
                  - "&7Line two average {average}"
                  - "&8[&n{action}&8]"
            """.trimIndent()
        )
        val p = player("Victim")
        verdicts(engine, p, 0.5)

        engine.punishPlayer(p)

        val captor = argumentCaptor<String>()
        // ban-with-ip: false — бан идёт через четырёхаргументный ban() (только профиль),
        // а не через banPlayerFull. E у ban()-method выведен из возвращаемого типа, поэтому
        // его нужно назвать явно.
        verify(p).ban<BanEntry<PlayerProfile>>(
            captor.capture(),
            Mockito.isNull<java.util.Date>(),
            captor.capture(),
            org.mockito.ArgumentMatchers.anyBoolean(),
        )
        assertEquals(
            "§6Line one 50.0%\n" +
                "§7Line two average 0.500\n" +
                "§8[§nban§8]",
            captor.firstValue,
        )
    }

    @Test
    fun `ban-ip message inherits ban template when key is absent`() {
        val engine = createEngine(
            """
            detection:
              punish-action: ban
              ban-with-ip: true
              punish-reason: test reason
              punish-message:
                ban:
                  - "&aShared text {action}"
            """.trimIndent()
        )
        val p = player("Victim")

        engine.punishPlayer(p)

        val captor = argumentCaptor<String>()
        verify(p).banPlayerFull(captor.capture())
        assertEquals("§aShared text ban-ip", captor.firstValue)
    }

    @Test
    fun `ban-ip message overrides ban when key is present`() {
        val base =
            "  punish-action: ban\n" +
            "  punish-reason: test reason\n" +
            "  punish-message:\n" +
            "    ban: \"&7plain ban {action}\"\n" +
            "    ban-ip: \"&4hard ban {action}\""

        val profileOnly = player("ProfileOnly")
        createEngine("detection:\n$base").punishPlayer(profileOnly)

        val profileCaptor = argumentCaptor<String>()
        // ban-with-ip: false → профильный ban(), а не banPlayerFull.
        verify(profileOnly).ban<BanEntry<PlayerProfile>>(
            profileCaptor.capture(),
            Mockito.isNull<java.util.Date>(),
            profileCaptor.capture(),
            org.mockito.ArgumentMatchers.anyBoolean(),
        )
        assertEquals("§7plain ban ban", profileCaptor.firstValue)

        val withIp = player("WithIp")
        createEngine("detection:\n  ban-with-ip: true\n$base").punishPlayer(withIp)

        val ipCaptor = argumentCaptor<String>()
        verify(withIp).banPlayerFull(ipCaptor.capture())
        assertEquals("§4hard ban ban-ip", ipCaptor.firstValue)
    }

    @Test
    fun `empty punish-message falls back to defaults`() {
        val engine = createEngine("detection:\n  punish-message:\n    kick: \"\"\n    ban: []")
        val p = player("Victim")

        engine.punishPlayer(p)

        val captor = argumentCaptor<String>()
        verify(p).kickPlayer(captor.capture())
        assertTrue(captor.firstValue.contains("сообщите администрации"))
    }

    @Test
    fun `punish message translates ampersand hex colors and passthrough`() {
        val engine = createEngine(
            """
            detection:
              punish-message:
                kick: "&#22761BHex&b&L&O K&#GGGGGG&#22761B"
            """.trimIndent()
        )
        val p = player("Victim")

        engine.punishPlayer(p)

        val captor = argumentCaptor<String>()
        verify(p).kickPlayer(captor.capture())
        assertEquals(
            "§x§2§2§7§6§1§bHex§b§l§o K&#GGGGGG§x§2§2§7§6§1§b",
            captor.firstValue,
            "hex-цвет переводится, невалидный &#GGGGGG остаётся текстом",
        )
    }

    @Test
    fun `offline player is not punished`() {
        val engine = createEngine(custom())
        val p = player()
        whenever(p.isOnline).thenReturn(false)
        engine.recordHit(p)

        engine.punishPlayer(p)

        verify(p, never()).kickPlayer(any<String>())
        assertEquals(1, engine.getHitCount(p.uniqueId), "tracking оффлайн-игрока не трогаем")
    }

    @Test
    fun `kick clears all tracking after punishment`() {
        val engine = createEngine(custom())
        val p = player("Victim")
        verdicts(engine, p, 0.5, 0.7)
        engine.recordHit(p)

        engine.punishPlayer(p)

        verify(p).kickPlayer(contains("test reason"))
        assertNull(engine.getAveragePrediction(p))
        assertTrue(engine.getTrackedPlayerIds().isEmpty())
        assertEquals(0, engine.getHitCount(p.uniqueId))
    }

    @Test
    fun `none action keeps tracking because player stays online`() {
        val engine = createEngine(custom(action = "none"))
        val p = player()
        verdicts(engine, p, 0.5)
        engine.recordHit(p)

        engine.punishPlayer(p)

        verify(p, never()).kickPlayer(any<String>())
        assertNotNull(engine.getAveragePrediction(p))
        assertEquals(1, engine.getHitCount(p.uniqueId))
        assertTrue(engine.getTrackedPlayerIds().contains(p.uniqueId))
    }

    @Test
    fun `kick falls back to console command on linkage error`() {
        val engine = createEngine(custom())
        val p = player("Victim")
        verdicts(engine, p, 0.5)
        doThrow(LinkageError("missing method")).whenever(p).kickPlayer(any<String>())

        engine.punishPlayer(p)

        bukkit.verify { Bukkit.dispatchCommand(any<CommandSender>(), eq("kick Victim test reason")) }
        assertEquals(0, engine.getHitCount(p.uniqueId), "после кика tracking всё равно чистится")
    }

    @Test
    fun `ip ban falls back to ban-ip console command`() {
        val engine = createEngine(custom(action = "ban", banWithIp = true))
        val p = player("Victim")
        verdicts(engine, p, 0.5)
        doThrow(LinkageError("missing method")).whenever(p).banPlayerFull(any<String>())

        engine.punishPlayer(p)

        bukkit.verify { Bukkit.dispatchCommand(any<CommandSender>(), eq("ban-ip Victim test reason")) }
    }

    // --- TTL-чистка ----------------------------------------------------------

    @Test
    fun `stale tracking is pruned after an hour`() {
        val engine = createEngine()
        val p = player()
        verdicts(engine, p, 0.8)
        engine.recordHit(p)

        now += 61 * 60 * 1000
        engine.pruneStaleTracking()

        assertTrue(engine.getTrackedPlayerIds().isEmpty())
        assertNull(engine.getAveragePrediction(p))
        assertEquals(0, engine.getHitCount(p.uniqueId))
    }

    @Test
    fun `fresh tracking survives pruning`() {
        val engine = createEngine()
        val p = player()
        verdicts(engine, p, 0.8)
        engine.recordHit(p)

        engine.pruneStaleTracking()

        assertEquals(0.8, engine.getAveragePrediction(p))
        assertEquals(1, engine.getHitCount(p.uniqueId))
    }

    @Test
    fun `hit count of an online player survives pruning`() {
        val engine = createEngine()
        val p = player()
        verdicts(engine, p, 0.8)
        engine.recordHit(p)
        bukkit.`when`<Player?> { Bukkit.getPlayer(p.uniqueId) }.thenReturn(p)

        now += 61 * 60 * 1000
        engine.pruneStaleTracking()

        assertTrue(engine.getTrackedPlayerIds().isEmpty(), "журнал всё равно чистится по TTL")
        assertEquals(1, engine.getHitCount(p.uniqueId), "онлайн-игрок остаётся в счётчиках")
    }
}
