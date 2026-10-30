package zov.grusha.daynAC.detection

import org.bukkit.Bukkit
import org.bukkit.entity.Player
import zov.grusha.daynAC.permissions.Permissions
import zov.grusha.daynAC.ml.FeatureExtractor
import zov.grusha.daynAC.ml.NeuralNetwork
import java.util.ArrayDeque
import java.util.Locale
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Движок детекции: связывает FeatureExtractor → NeuralNetwork → реакция сервера.
 *
 * Inference выполняется асинхронно в пуле daemon-потоков — тяжёлый прямой проход
 * (343 → 128 → 64 → 32 → 1) не блокирует основной тик сервера. Решения о наказании
 * и модификации урона принимаются только на главном потоке через scheduler.
 *
 * Защита от ложных срабатываний — двойная:
 *  1. Множитель урона считается по ПРЕДЫДУЩИМ предсказаниям: первый удар всегда
 *     пропускается полностью (нет данных — нет наказания).
 *  2. Автонаказание требует [punishConfirmWindows] НЕЗАВИСИМЫХ окон подряд со
 *     скором ≥ punishThreshold. Окна считаются на каждый удар и перекрываются
 *     на windowSize-1 ударов, поэтому «среднее по последним N предсказаниям»
 *     — это почти один и тот же срез данных, а не N подтверждений. Независимыми
 *     считаются только окна, между концами которых не меньше windowSize ударов
 *     (они не имеют общих ударов). См. [updatePunishProgress].
 */
class DetectionEngine(
    private val plugin: org.bukkit.plugin.java.JavaPlugin,
    private val network: NeuralNetwork,
    /** Размер окна ударов (DaynAC.WINDOW_SIZE): через столько ударов окна перестают пересекаться. */
    private val windowSize: Int,
    /**
     * Источник текущего времени (мс). В production — [System.currentTimeMillis];
     * в тестах подменяется, чтобы проверять простой, затухание carry-over и TTL
     * без Thread.sleep.
     */
    private val clock: () -> Long = { System.currentTimeMillis() },
    /**
     * Пул для тяжёлого predict. Тестовый шов: в тестах подставляется прямой
     * исполнитель, чтобы вердикт считался синхронно в вызывающем потоке.
     * В production — два daemon-потока, как и раньше.
     */
    private val inferencePool: ExecutorService = Executors.newFixedThreadPool(2) { runnable ->
        Thread(runnable, "DaynAC-Inference").apply { isDaemon = true }
    }
) {

    /**
     * Пороги детекции — читаются из config.yml (секция detection), значения
     * по умолчанию соответствуют daynac.md. Публичные val, чтобы /daynac info
     * и GUI показывали актуальные значения.
     */
    val flagThreshold: Double
    val reduceThreshold: Double
    val cancelThreshold: Double
    val punishThreshold: Double
    val maxRecentPredictions: Int

    /**
     * Сколько независимых (непересекающихся) окон подряд должны дать скор ≥ [punishThreshold],
     * чтобы сработало автонаказание. 1 = достаточно одного окна. Читается из
     * detection.punish-confirm-windows.
     */
    val punishConfirmWindows: Int

    /**
     * Кулдаун оповещений об одном игроке, секунды (0 = оповещение на каждый удар
     * ≥ [flagThreshold] — прежнее поведение, заметный спам при высокой CPS). Первый
     * подозрительный удар оповещает мгновенно; удары, пришедшие внутри кулдауна,
     * не теряются — суммируются в следующее оповещение. Кулдаун трогает только
     * сообщения: снижение урона, punish-progress и наказание работают без изменений.
     */
    val alertCooldownSeconds: Int

    /**
     * Простой в секундах, после которого с игрока снимаются флаги и окно ударов
     * начинается заново (0 = сброс выключен). См. [isIdle] и [resetFlags].
     */
    val inactivityResetSeconds: Int

    /**
     * Сколько секунд после сброса по простою помнить подозрительный скор игрока
     * (0 = не помнить). Пока новое окно ударов не набралось, урон считается по этому
     * затухающему скору, а не по «чистому листу». Читается из detection.carry-over-seconds.
     */
    val carryOverSeconds: Int

    /** Автонаказание (кик) включено? false = режим наблюдения: только оповещения и снижение урона. */
    val punishEnabled: Boolean

    /** Тип наказания: "none" | "kick" | "ban". */
    val punishAction: String

    /** Причина наказания для кика/бана. */
    val punishReason: String

    /** Банить ли одновременно по IP (punish-action: ban). true = бан профиля + бан IP. */
    val banWithIp: Boolean

    /**
     * Шаблоны сообщений наказанному игроку из detection.punish-message — то, что видит
     * игрок на экране кика и в бан-листе. Каждый элемент списка — отдельная строка экрана;
     * плейсхолдеры ({reason} {player} {score} {average} {action}) подставляются в
     * [punishPlayer], &-коды (включая hex &#RRGGBB) переводятся в §-формат.
     * Ключ ban-ip при отсутствии наследует текст ban (см. читателю config.yml).
     */
    val punishMessageKick: List<String>
    val punishMessageBan: List<String>
    val punishMessageBanIp: List<String>

    init {
        val config = plugin.config
        flagThreshold = config.getDouble("detection.flag-threshold", 0.65)
        reduceThreshold = config.getDouble("detection.reduce-threshold", 0.70)
        cancelThreshold = config.getDouble("detection.cancel-threshold", 0.90)
        punishThreshold = config.getDouble("detection.punish-threshold", 0.95)
        maxRecentPredictions = config.getInt("detection.max-recent-predictions", 10)
        punishConfirmWindows = config.getInt("detection.punish-confirm-windows", 2)
        alertCooldownSeconds = config.getInt("detection.alert-cooldown-seconds", 5)
        inactivityResetSeconds = config.getInt("detection.inactivity-reset-seconds", 30)
        carryOverSeconds = config.getInt("detection.carry-over-seconds", 600)
        punishEnabled = config.getBoolean("detection.punish-enabled", true)
        punishAction = config.getString("detection.punish-action", "kick")?.lowercase() ?: "kick"
        punishReason = config.getString("detection.punish-reason", "KillAura (обнаружено нейросетью DaynAC)")!!
        banWithIp = config.getBoolean("detection.ban-with-ip", false)
        punishMessageKick = readMessageTemplate(config, "detection.punish-message.kick")
            ?: DEFAULT_PUNISH_MESSAGE_KICK
        punishMessageBan = readMessageTemplate(config, "detection.punish-message.ban")
            ?: DEFAULT_PUNISH_MESSAGE_BAN
        // Наследование из config.yml: «Ключ ban-ip наследует ban, если не задан».
        punishMessageBanIp = readMessageTemplate(config, "detection.punish-message.ban-ip")
            ?: punishMessageBan

        require(punishAction in setOf("none", "kick", "ban")) {
            "detection.punish-action должен быть none/kick/ban, получено: \"$punishAction\""
        }

        // Санитарная проверка: пороги вне (0, 1) или в абсурдном порядке ломают логику.
        // Инвариант из config.yml: flag ≤ reduce < cancel ≤ punish — без него,
        // например, punish < flag означал бы автокик без единого оповещения админам.
        require(flagThreshold in 0.0..1.0) { "detection.flag-threshold должен быть в (0, 1), получено $flagThreshold" }
        require(reduceThreshold in 0.0..1.0) { "detection.reduce-threshold должен быть в (0, 1), получено $reduceThreshold" }
        require(cancelThreshold in 0.0..1.0) { "detection.cancel-threshold должен быть в (0, 1), получено $cancelThreshold" }
        require(punishThreshold in 0.0..1.0) { "detection.punish-threshold должен быть в (0, 1), получено $punishThreshold" }
        require(flagThreshold <= reduceThreshold) {
            "detection.flag-threshold ($flagThreshold) должен быть ≤ reduce-threshold ($reduceThreshold), иначе наказание срабатывает без оповещения"
        }
        require(cancelThreshold > reduceThreshold) {
            "detection.cancel-threshold ($cancelThreshold) должен быть больше reduce-threshold ($reduceThreshold)"
        }
        require(cancelThreshold <= punishThreshold) {
            "detection.cancel-threshold ($cancelThreshold) должен быть ≤ punish-threshold ($punishThreshold)"
        }
        require(carryOverSeconds >= 0) { "detection.carry-over-seconds должен быть ≥ 0 (0 = выключен), получено $carryOverSeconds" }
        require(punishConfirmWindows >= 1) { "detection.punish-confirm-windows должен быть ≥ 1, получено $punishConfirmWindows" }
        require(alertCooldownSeconds >= 0) { "detection.alert-cooldown-seconds должен быть ≥ 0 (0 = без кулдауна), получено $alertCooldownSeconds" }
        require(windowSize >= 1) { "windowSize должен быть ≥ 1, получено $windowSize" }
        require(maxRecentPredictions >= 1) { "detection.max-recent-predictions должен быть ≥ 1, получено $maxRecentPredictions" }
        require(inactivityResetSeconds >= 0) { "detection.inactivity-reset-seconds должен быть ≥ 0 (0 = выключен), получено $inactivityResetSeconds" }
    }

    private data class PlayerVerdict(
        val probability: Double,
        val timestamp: Long
    )

    /** Записи о неактивных игроках старше этого возраста удаляются (TTL против монотонного роста карт). */
    private val trackedTtlMs = 60L * 60L * 1000L // 1 час

    /**
     * Длина журнала скоров ([scoreHistory]). Заметно больше окна усреднения:
     * журнал — витрина для админа («Последние удары», «Макс»), а не вход для
     * решений, и должен переживать сброс по простою.
     */
    private val maxScoreHistory = 100

    /** Цветовые коды Minecraft — вычищаются, когда оповещение пишется в консольный лог. */
    private val colorCodes = Regex("§[0-9a-f]")

    /**
     * АКТИВНЫЕ предсказания по игрокам — то, чем движок реально руководствуется:
     * множитель урона, оповещения, автонаказание. Сбрасываются по простою
     * ([resetFlags]), наказанием и TTL-чисткой.
     */
    private val predictions = ConcurrentHashMap<UUID, ArrayDeque<PlayerVerdict>>()

    /**
     * Журнал ВСЕХ скоров игрока — то, что видит админ в /daynac allinfo
     * («Последние удары», «Макс»). Сброс по простою его НЕ трогает: с игрока
     * снимаются флаги, но история ударов остаётся для разбора. Чистится только
     * наказанием и TTL-чисткой.
     */
    private val scoreHistory = ConcurrentHashMap<UUID, ArrayDeque<PlayerVerdict>>()

    /**
     * Сколько ударов игрок сделал за всё время (GUI показывает это как «Всего ударов»).
     * Для прогресса ТЕКУЩЕГО окна (и подсказки «нужно 16» в /daynac ver) используется
     * [zov.grusha.daynAC.tracking.HitTracker.getWindowFill]: после сброса по простою
     * этот счётчик СОЗНАТЕЛЬНО не обнуляется — он хранит историческую статистику.
     */
    val hitCounts = java.util.concurrent.ConcurrentHashMap<UUID, Int>()

    /**
     * Время последнего удара игрока (мс). По нему считается простой: игрок,
     * не бивший никого дольше [inactivityResetSeconds], начинает с чистого листа.
     * Заполняется в [recordHit], очищается в [resetFlags] и [punishPlayer].
     */
    private val lastHitTimes = ConcurrentHashMap<UUID, Long>()

    /**
     * Прогресс подтверждения наказания: сколько независимых окон подряд уже
     * дали скор ≥ punishThreshold и сколько вердиктов прошло с последнего из них.
     * Трогается только на главном потоке (handlePrediction).
     */
    private class PunishProgress {
        var confirmed: Int = 0
        var verdictsSinceConfirm: Int = 0
    }

    private val punishProgress = ConcurrentHashMap<UUID, PunishProgress>()

    /**
     * Состояние кулдауна оповещений игрока: когда ушло последнее сообщение и что
     * накопилось с тех пор (сколько необычных ударов подавлено, их результат).
     * Трогается только на главном потоке (handlePrediction, resetFlags →
     * flushPendingAlert).
     */
    private class AlertState {
        var lastAlertTimeMillis: Long = 0L // 0 = ещё не оповещали
        var suppressedEvents: Int = 0
        var suppressedMaxScore: Double = 0.0
    }

    private val alertStates = ConcurrentHashMap<UUID, AlertState>()

    /** Скор игрока на момент сброса по простою и время сброса — см. [carryOverSeconds]. */
    private class CarryOver(val score: Double, val timeMillis: Long)

    private val carryOver = ConcurrentHashMap<UUID, CarryOver>()

    /**
     * Сколько ударов после контрольной точки независимое окно ещё может «подтвердиться».
     * Без запаса один вердикт 0.94 ровно в контрольной точке обнулял серию даже у явного
     * читера, чьи скоры держатся около порога (0.95–1.00).
     */
    private val confirmGrace: Int = windowSize / 2

    /** Модель обучена и загружена — детекция активна. */
    @Volatile
    var detectionEnabled: Boolean = false
        private set

    fun enableDetection() {
        detectionEnabled = true
    }

    fun disableDetection() {
        detectionEnabled = false
    }

    /** Регистрирует удар игрока (вызывается на главном потоке при каждом ударе). */
    fun recordHit(player: Player) {
        hitCounts.merge(player.uniqueId, 1, Int::plus)
        lastHitTimes[player.uniqueId] = clock()
    }

    /** Сколько ударов у игрока и хватает ли их для анализа (окно 16). */
    fun getHitCount(playerId: UUID): Int = hitCounts[playerId] ?: 0

    /**
     * Прошло ли с последнего удара игрока больше [inactivityResetSeconds].
     * Игрок без единого удара или уже сброшенный простаивающим не считается —
     * сбрасывать нечего.
     */
    fun isIdle(playerId: UUID): Boolean {
        if (inactivityResetSeconds <= 0) return false
        val lastHit = lastHitTimes[playerId] ?: return false
        return clock() - lastHit > inactivityResetSeconds * 1000L
    }

    /**
     * Удаляет записи о неактивных игроках старше TTL. Предсказания и счётчики
     * ударов сохраняются после выхода игрока специально (GUI показывает и
     * оффлайн-игроков), но без ограничения срока они копятся бесконечно на
     * долгоживущем сервере. Вызывается периодически из onEnable — независимо
     * от того, открывает ли кто-то GUI.
     */
    fun pruneStaleTracking() {
        val now = clock()
        predictions.entries.removeIf { (_, deque) ->
            val lastTs = synchronized(deque) { deque.lastOrNull()?.timestamp }
            lastTs == null || now - lastTs > trackedTtlMs
        }
        // Журнал чистится по тому же TTL: админ видит историю игрока час после
        // последнего удара, дальше запись забывается вместе с активными флагами.
        scoreHistory.entries.removeIf { (_, deque) ->
            val lastTs = synchronized(deque) { deque.lastOrNull()?.timestamp }
            lastTs == null || now - lastTs > trackedTtlMs
        }
        // Счётчик ударов мал сам по себе; удаляем только записи оффлайн-игроков,
        // которых уже нет ни в журнале, ни в предсказаниях (иначе GUI потеряет
        // «прогресс окна»).
        hitCounts.keys.retainAll { uuid ->
            scoreHistory.containsKey(uuid) || predictions.containsKey(uuid) || Bukkit.getPlayer(uuid) != null
        }
        // Время последнего удара нужно только тем, кого мы ещё отслеживаем:
        // для забытого игрока (оффлайн, без предсказаний и ударов) хранить его нечего.
        lastHitTimes.keys.retainAll { uuid ->
            scoreHistory.containsKey(uuid) || predictions.containsKey(uuid) ||
                    hitCounts.containsKey(uuid) || Bukkit.getPlayer(uuid) != null
        }
        // Кулдаун оповещений чистится вместе с прочим отслеживанием: у забытого
        // игрока (оффлайн, без предсказаний и истории) состояние не нужно. Сброс
        // по простою состояние переживает — scoreHistory его «держит» живым.
        alertStates.keys.retainAll { uuid ->
            scoreHistory.containsKey(uuid) || predictions.containsKey(uuid) || Bukkit.getPlayer(uuid) != null
        }
        punishProgress.keys.retainAll { uuid -> predictions.containsKey(uuid) }
        carryOver.entries.removeIf { now - it.value.timeMillis >= carryOverSeconds * 1000L }
    }

    /**
     * Асинхронный анализ удара. Вектор признаков уже извлечён на главном потоке
     * (это дёшево), тяжёлый predict уходит в пул, результат возвращается на
     * главный поток для оповещения/наказания.
     */
    fun analyzeHitAsync(attacker: Player, vector: FeatureExtractor.FeatureVector, onVerdict: (Player, Double) -> Unit) {
        val featuresCopy = vector.features.copyOf() //snapshot для асинхронного потока

        inferencePool.execute {
            val probability = try {
                network.predict(featuresCopy)
            } catch (e: Exception) {
                plugin.logger.warning("Inference failed: ${e.message}")
                return@execute
            }

            if (!probability.isFinite()) return@execute

            // Обратно на главный поток — трогать Bukkit API из пула нельзя.
            // После onDisable планировщик отвергает задачи (IllegalPluginAccessException),
            // а inference-потоки daemon и могут закончить работу во время shutdown.
            if (plugin.isEnabled) {
                Bukkit.getScheduler().runTask(plugin, Runnable {
                    if (!attacker.isOnline) return@Runnable
                    recordVerdict(attacker, probability)
                    onVerdict(attacker, probability)
                })
            }
        }
    }

    private fun recordVerdict(attacker: Player, probability: Double) {
        val verdict = PlayerVerdict(probability, clock())

        val deque = predictions.computeIfAbsent(attacker.uniqueId) { ArrayDeque() }
        synchronized(deque) {
            if (deque.size >= maxRecentPredictions) deque.removeFirst()
            deque.addLast(verdict)
        }

        // Тот же скор — в журнал. Он длиннее окна усреднения: GUI показывает
        // «Последние удары» и «Макс» за всю историю, а не только за активное окно.
        val history = scoreHistory.computeIfAbsent(attacker.uniqueId) { ArrayDeque() }
        synchronized(history) {
            if (history.size >= maxScoreHistory) history.removeFirst()
            history.addLast(verdict)
        }
    }

    /**
     * Множитель урона по ПРЕДЫДУЩИМ предсказаниям (вызывается синхронно на главном
     * потоке до завершения текущего асинхронного анализа):
     *   p < reduceThreshold          → 1.0 (полный урон)
     *   reduce ≤ p < cancel          → 1 / (1.2 + t·0.3), t = (p-reduce)/(cancel-reduce)
     *   p ≥ cancelThreshold          → 0.0 (полная отмена)
     */
    fun getDamageMultiplier(attacker: Player): Double {
        // Активное окно ударов главнее; пока вердиктов нет (новый бой после простоя),
        // берём затухающий скор с прошлого раза.
        val p = getAveragePrediction(attacker) ?: carriedScore(attacker.uniqueId) ?: return 1.0

        return when {
            p >= cancelThreshold -> 0.0
            p >= reduceThreshold -> {
                val t = (p - reduceThreshold) / (cancelThreshold - reduceThreshold)
                1.0 / (1.2 + t * 0.3)
            }
            else -> 1.0
        }
    }

    /**
     * Скор с момента сброса по простою, линейно затухающий до 0 за [carryOverSeconds].
     * null — нечего помнить или всё затухло.
     */
    private fun carriedScore(playerId: UUID): Double? {
        if (carryOverSeconds <= 0) return null
        val carried = carryOver[playerId] ?: return null
        val elapsed = clock() - carried.timeMillis
        val lifetime = carryOverSeconds * 1000L
        if (elapsed >= lifetime) {
            carryOver.remove(playerId)
            return null
        }
        return carried.score * (1.0 - elapsed.toDouble() / lifetime)
    }

    /** Среднее по последним предсказаниям игрока (или null, если их ещё нет). */
    fun getAveragePrediction(player: Player): Double? {
        val deque = predictions[player.uniqueId] ?: return null
        val snapshot = synchronized(deque) { deque.toList() }
        if (snapshot.isEmpty()) return null
        return snapshot.sumOf { it.probability } / snapshot.size
    }

    /** Последнее предсказание (для /daynac ver). */
    fun getLastPrediction(player: Player): Double? =
        predictions[player.uniqueId]?.let { deque -> synchronized(deque) { deque.lastOrNull()?.probability } }

    /**
     * Последние АКТИВНЫЕ предсказания игрока (новые в конце) — окно усреднения.
     * После сброса по простою пусто, пока игрок не наберёт окно ударов заново;
     * сводка в оповещении админам строится именно по ним.
     */
    fun getRecentPredictions(playerId: UUID): List<Double> {
        val deque = predictions[playerId] ?: return emptyList()
        return synchronized(deque) { deque.map { it.probability } }
    }

    /** Журнал скоров игрока (новые в конце) — «Последние удары» в GUI. Сброс по простою его не трогает. */
    fun getScoreHistory(playerId: UUID): List<Double> {
        val deque = scoreHistory[playerId] ?: return emptyList()
        return synchronized(deque) { deque.map { it.probability } }
    }

    /** Максимальный скор за всю историю наблюдений игрока (журнал, не окно). */
    fun getMaxPrediction(playerId: UUID): Double? =
        getScoreHistory(playerId).maxOrNull()

    /**
     * Все UUID, по которым есть хоть одно предсказание (для GUI). Берём журнал,
     * а не активные предсказания: после сброса по простою он пуст, но игрок
     * должен остаться в /daynac allinfo.
     */
    fun getTrackedPlayerIds(): Set<UUID> = scoreHistory.keys.toSet()

    /**
     * Проверка порогов на главном потоке: оповещение админов при подозрении,
     * автонаказание при устойчиво высоком скоре.
     * @return true, если игрока нужно наказать (кик/бан — решает вызывающий код)
     */
    fun handlePrediction(attacker: Player, probability: Double): Boolean {
        val average = getAveragePrediction(attacker) ?: probability
        val confirmedBefore = punishProgress[attacker.uniqueId]?.confirmed ?: 0
        val progress = updatePunishProgress(attacker.uniqueId, probability)
        val seriesGrew = progress.confirmed > confirmedBefore
        // Байпас кулдауна заслуживает только предвестие РЕАЛЬНОГО наказания:
        // засчитано последнее окно серии, кик/бан сейчас произойдёт. Байпас на
        // любой рост (включая открытие 0→1) недопустим: серия «расходуется»
        // в handlePrediction независимо от punish-enabled и punish-action, поэтому
        // в режимах наблюдения (punish-enabled: false, punish-action: none)
        // новая серия открывается каждые ~windowSize ударов, и байпас возвращал
        // бы спам (с punish-confirm-windows: 1 — на каждый удар ≥ punishThreshold).
        // Там кулдаун обязателен для всех сообщений.
        val isEscalation = punishEnabled && punishAction != "none" &&
                seriesGrew && progress.confirmed >= punishConfirmWindows

        // Оповещение об ударе ≥ flag-threshold, но не чаще alert-cooldown-seconds на
        // игрока: вердикты приходят на каждый удар (окна перекрываются), и без кулдауна
        // подозрительный игрок сыпал бы по несколько сообщений в секунду в чат админов
        // и консоль. Первый удар кулдаун-окна оповещается мгновенно; остальные за окно
        // не теряются — идут агрегатом в следующее сообщение. Предвестие реального
        // наказания (isEscalation) и сам факт наказания (уведомление в punishPlayer)
        // идут мимо кулдауна. На детекцию кулдаун не влияет: punish-progress обновлён
        // выше, наказание работает как раньше.
        if (probability >= flagThreshold) {
            buildAlert(attacker, probability, average, progress, isEscalation)?.let { message ->
                sendAlert(message)
            }
        }

        // Автонаказание: нужны punishConfirmWindows независимых окон подряд ≥ punishThreshold.
        // Серия «расходуется» при срабатывании (в том числе в режиме наблюдения и при
        // punish-action: none) — повторное срабатывание потребует новой полной серии,
        // а не будет повторяться на каждом ударе.
        val confirmed = progress.confirmed >= punishConfirmWindows
        if (confirmed) punishProgress.remove(attacker.uniqueId)
        return punishEnabled && confirmed
    }

    /**
     * Кулдаун оповещений: первый удар ≥ flag-threshold за кулдаун-интервал
     * отправляется немедленно, остальные за интервал подавляются, запоминаются
     * (количество и максимальный скор) и агрегируются в следующее сообщение.
     * [isEscalation] — засчитано последнее окно серии перед реальным наказанием
     * (punish-enabled и punish-action ≠ none): такое сообщение идёт мимо кулдауна
     * и выпускает накопленный агрегат. Рост серии БЕЗ наказания (режимы
     * наблюдения, где серия расходуется и открывается заново) подчиняется
     * кулдауну наравне с обычными оповещениями — иначе цикличные открытия
     * возвращали бы спам.
     *
     * Возвращает null, если удар попал под кулдаун: текст не отправлен, но сама
     * детекция уже сработала — punish-progress в handlePrediction обновлён.
     */
    private fun buildAlert(
        attacker: Player,
        probability: Double,
        average: Double,
        progress: PunishProgress,
        seriesGrew: Boolean,
    ): String? {
        if (alertCooldownSeconds <= 0) {
            // Кулдаун выключен: сообщение на каждый удар, как и раньше.
            return formatAlert(attacker, probability, average, progress, 0, 0.0, 0)
        }
        val state = alertStates.computeIfAbsent(attacker.uniqueId) { AlertState() }
        val now = clock()
        // lastAlertTimeMillis == 0L — «оповещений ещё не было», это НЕ кулдаун.
        // Явная проверка нужна, чтобы на часах, отсчитывающих время от малого
        // значения (тестовый шов clock), первое оповещение не подавилось.
        val cooldownPassed = state.lastAlertTimeMillis == 0L ||
                now - state.lastAlertTimeMillis >= alertCooldownSeconds * 1000L
        if (!cooldownPassed && !seriesGrew) {
            state.suppressedEvents++
            state.suppressedMaxScore = maxOf(state.suppressedMaxScore, probability)
            return null
        }
        val suppressedEvents = state.suppressedEvents
        val suppressedMaxScore = state.suppressedMaxScore
        val sinceLastAlertSeconds =
            if (state.lastAlertTimeMillis > 0L) ((now - state.lastAlertTimeMillis) / 1000L).toInt() else 0
        state.lastAlertTimeMillis = now
        state.suppressedEvents = 0
        state.suppressedMaxScore = 0.0
        return formatAlert(attacker, probability, average, progress, suppressedEvents, suppressedMaxScore, sinceLastAlertSeconds)
    }

    /**
     * Рассылает оповещение обладателям daynac.alerts и пишет очищенный от цветовых
     * кодов текст в консольный лог. log=false — для уведомлений, у которых своя
     * консольная строка уже есть (наказание логируется ветками punishPlayer).
     */
    private fun sendAlert(message: String, log: Boolean = true) {
        Bukkit.getOnlinePlayers().filter { it.hasPermission(Permissions.ALERTS) }.forEach { it.sendMessage(message) }
        if (log) plugin.logger.warning(message.replace(colorCodes, ""))
    }

    /**
     * Итоговый выпуск агрегата подавленных кулдауном ударов при окончании боя —
     * сброс по простою, наказание не случилось или ручная очистка флагов. Без
     * этого suppressedEvents жили до следующего подозрительного удара, а тот мог
     * не наступить: подавленная активность пропадала бы из поля зрения админов
     * навсегда (состояние умирает в TTL-чистке). После выпуска кулдаун снимается
     * целиком — первый подозрительный удар нового боя оповещает мгновенно.
     */
    private fun flushPendingAlert(playerId: UUID) {
        val state = alertStates[playerId] ?: return
        val pendingEvents = state.suppressedEvents
        val pendingMaxScore = state.suppressedMaxScore
        state.suppressedEvents = 0
        state.suppressedMaxScore = 0.0
        state.lastAlertTimeMillis = 0L
        if (pendingEvents == 0) return
        // Имя нужно для текста; не в сети (например, цель ручной очистки) —
        // выпускать некуда, состояние по-прежнему очищено.
        val name = Bukkit.getPlayer(playerId)?.name ?: return
        sendAlert(
            "[DaynAC] §c$name§7: подозрительная серия окончена — ещё §c$pendingEvents§7 необычных ударов не показано, макс §e" +
                    String.format(Locale.US, "%.2f", pendingMaxScore) + "§7"
        )
    }

    /**
     * Текст оповещения: текущий удар, динамика последних 4 хитов, среднее,
     * прогресс окон наказания и агрегат ударов, не показанных из-за кулдауна.
     */
    private fun formatAlert(
        attacker: Player,
        probability: Double,
        average: Double,
        progress: PunishProgress,
        suppressedEvents: Int,
        suppressedMaxScore: Double,
        sinceLastAlertSeconds: Int,
    ): String = buildString {
        append("[DaynAC] §c")
        append(attacker.name)
        append("§7: необычный удар §e")
        append(String.format(Locale.US, "%.3f", probability))
        append("§7 > порога §f")
        append(String.format(Locale.US, "%.2f", flagThreshold))
        append("§7 — последние удары: ")
        // Сводка по последним 4 ударам — динамика вокруг необычного удара
        getRecentPredictions(attacker.uniqueId).takeLast(4).joinTo(this, " ") { score ->
            scoreColor(score) + String.format(Locale.US, "%.2f", score)
        }
        append("§7 (средний: §e")
        append(String.format(Locale.US, "%.3f", average))
        append("§7)")
        if (progress.confirmed > 0) {
            val status = when {
                progress.confirmed >= punishConfirmWindows -> "§cнаказание"
                progress.verdictsSinceConfirm >= windowSize -> "§7проверка окна сейчас"
                else -> "§7следующее окно через ${windowSize - progress.verdictsSinceConfirm} уд."
            }
            append(" §8[окна для наказания: §c${progress.confirmed}§8/$punishConfirmWindows§8, $status§8]")
        }
        if (suppressedEvents > 0) {
            // Удары, скрытые кулдауном, показываются как агрегат, чтобы картина
            // не занижала активность читера. С байпасом агрегат может уйти
            // в тот же тик, когда накопился (sinceLastAlertSeconds == 0) —
            // показываем «<1 с» честно.
            append(" §8[ещё §c$suppressedEvents§8 уд. за §f")
            append(if (sinceLastAlertSeconds <= 0) "<1 с" else "$sinceLastAlertSeconds с")
            append("§8, макс §e")
            append(String.format(Locale.US, "%.2f", suppressedMaxScore))
            append("§8]")
        }
    }

    /**
     * Обновляет серию подтверждений для игрока по свежему вердикту.
     *
     * Вердикт приходит на КАЖДЫЙ удар, а окно признаков — последние windowSize ударов,
     * поэтому соседние вердикты почти не независимы. Независимым подтверждением
     * считаем только вердикт, пришедший ровно через windowSize ударов после
     * предыдущего засчитанного: его окно не имеет общих ударов с предыдущим.
     *
     *  - серии нет: первый вердикт ≥ punishThreshold открывает её (confirmed = 1);
     *  - серия идёт: считаем вердикты; когда прошло windowSize, окно независимо от
     *    предыдущего. Первый же вердикт ≥ punishThreshold засчитывает его. Если вердикты
     *    ниже порога, ждём ещё до [confirmGrace] ударов и только потом рвём серию.
     *    Вердикты между контрольными точками серию не меняют (они перекрываются).
     *
     * Если вердикт не дошёл (ошибка inference, игрок вышел), счётчик отстаёт от числа
     * ударов, то есть интервал между окнами в ударах только больше — безопасно.
     */
    private fun updatePunishProgress(playerId: UUID, probability: Double): PunishProgress {
        val progress = punishProgress.computeIfAbsent(playerId) { PunishProgress() }
        if (progress.confirmed == 0) {
            if (probability >= punishThreshold) {
                progress.confirmed = 1
                progress.verdictsSinceConfirm = 0
            }
        } else {
            progress.verdictsSinceConfirm++
            if (progress.verdictsSinceConfirm >= windowSize) {
                // Окно уже независимо от предыдущего засчитанного. Ждём вердикта ≥ порога
                // до windowSize + confirmGrace ударов; не дождались — окно чистое, серия прервана.
                if (probability >= punishThreshold) {
                    progress.confirmed++
                    progress.verdictsSinceConfirm = 0
                } else if (progress.verdictsSinceConfirm >= windowSize + confirmGrace) {
                    progress.confirmed = 0
                    progress.verdictsSinceConfirm = 0
                }
            }
        }
        return progress
    }

    /** Цвет числа скора в сводке — по тем же диапазонам, что и иконки GUI. */
    private fun scoreColor(score: Double): String = when {
        score < 0.4 -> "§a"
        score < 0.65 -> "§e"
        score < 0.9 -> "§c"
        else -> "§4"
    }

    /**
     * Читает шаблон сообщения из detection.punish-message.<ключ>. Допустимы одна
     * строка (переносы \n остаются внутри неё) и YAML-список строк (каждая — новая
     * строка экрана, как описано в комментарии к конфигу). Отсутствующий или пустой
     * ключ даёт null — вызывающий код берёт значение по умолчанию (для ban-ip это
     * текст ban: наследование в конфиге описано).
     */
    private fun readMessageTemplate(
        config: org.bukkit.configuration.file.FileConfiguration,
        path: String,
    ): List<String>? {
        val value = config.get(path) ?: return null
        val lines = when (value) {
            is String -> listOf(value)
            is List<*> -> value.map { it.toString() }
            else -> null
        } ?: return null
        // Полностью пустой шаблон бессмысленен — считаем ключ не заданным (откат к
        // дефолту или наследованию ban). Одиночные пустые строки внутри списка
        // оставляем: это сознательные «пустые» строки на экране наказанного.
        return if (lines.isEmpty() || lines.all { it.isBlank() }) null else lines
    }

    /**
     * Собирает сообщение наказания из шаблона: подставляет плейсхолдеры, соединяет
     * строки переносом и переводит &-коды в §-формат ([translateColorCodes]).
     *
     * Плейсхолдеры (см. комментарии к punish-message в config.yml):
     *   {reason}  — punish-reason
     *   {player}  — ник наказанного
     *   {score}   — средняя вероятность с процентом, например 97.3%
     *   {average} — она же числом, например 0.973
     *   {action}  — kick | ban | ban-ip
     *
     * Подстановка идёт ДО перевода цветов: значения (например, свой punish-reason)
     * могут сами содержать &-коды и после переподстановки тоже окрашиваются.
     */
    private fun renderPunishMessage(
        template: List<String>,
        player: Player,
        average: Double,
        action: String,
    ): String {
        val placeholders = mapOf(
            "reason" to punishReason,
            "player" to player.name,
            "score" to String.format(Locale.US, "%.1f", average * 100) + "%",
            "average" to String.format(Locale.US, "%.3f", average),
            "action" to action,
        )
        val rendered = template.map { line ->
            placeholders.entries.fold(line) { text, (key, value) -> text.replace("{$key}", value) }
        }
        return rendered.joinToString("\n") { translateColorCodes(it) }
    }

    /**
     * Перевод цветовых кодов в legacy-формат, который принимает kickPlayer и бан-лист:
     *   &[0-9a-fk-or] → §код (&-код допустим и в заглавных, вывод — строчные)
     *   &#RRGGBB      → §x§R§R§G§G§B§B — ванильный способ записать hex-цвет
     *                    семью обычными константами (клиенты с 1.16).
     * Готовые §-коды не трогаются: конфиги со старым § работают как раньше.
     */
    private fun translateColorCodes(text: String): String {
        if ('&' !in text) return text
        val sb = StringBuilder(text.length)
        var i = 0
        while (i < text.length) {
            val ch = text[i]
            if (ch == '&' && i + 1 < text.length) {
                val code = text[i + 1]
                // &#RRGGBB — после '#' нужно ещё 6 символов.
                if (code == '#' && i + 8 <= text.length) {
                    val hex = text.substring(i + 2, i + 8)
                    if (HEX_COLOR.matches(hex)) {
                        sb.append("§x")
                        hex.lowercase().forEach { c -> sb.append('§').append(c) }
                        i += 8
                        continue
                    }
                }
                if (code.lowercaseChar() in LEGACY_CODE_CHARS) {
                    sb.append('§').append(code.lowercaseChar())
                    i += 2
                    continue
                }
            }
            // Не код — '&' идёт на экран как обычный символ; то же для '#' рядом
            // с ним: "&#GGGGGG" (не шесть цифр) ключом не считается.
            sb.append(ch)
            i++
        }
        return sb.toString()
    }

    /**
     * Наказание согласно detection.punish-action из конфига:
     *  none — только запись в лог (полное бездействие)
     *  kick — кик с сообщением
     *  ban  — бан; при detection.ban-with-ip=true дополнительно бан по IP
     *         (banPlayerFull), иначе только профиль (UUID)
     *
     * Текст, который видит наказанный (и который хранится в бан-листе), собирается
     * из шаблонов detection.punish-message через [renderPunishMessage]: kick —
     * из punish-message.kick, ban — из punish-message.ban, ban + IP — из
     * punish-message.ban-ip (наследует ban, если ключ не задан). Запасные
     * консольные ban/kick по-прежнему передают punish-reason одной строкой.
     *
     * При "none" игрок остаётся на сервере, поэтому tracking-карты НЕ трогаем.
     * predictions — это та самая история, по которой handlePrediction считает
     * среднее для второй ступени защиты; стерев её, получаем avg == мгновенному
     * скору, из-за чего наказание срабатывает на каждом ударе ≥ punishThreshold,
     * а окно навсегда остаётся из одного элемента (в алерте: «1.00 (средний: 1.000)»).
     *
     * После кика/бана игрок удаляется из tracking-карт: дело закрыто,
     * из /daynac allinfo он пропадает (иначе наказанный висел бы в GUI
     * как «Не в сети» до истечения TTL).
     */
    fun punishPlayer(player: Player) {
        if (!player.isOnline) return
        val average = getAveragePrediction(player) ?: 0.0
        val percent = String.format(Locale.US, "%.1f", average * 100)

        when (punishAction) {
            "none" -> {
                plugin.logger.warning("[DaynAC] Наказание отключено (punish-action: none): ${player.name}, средний скор $average")
                // Игрок остался на сервере: историю предсказаний сохраняем —
                // по ней считается среднее для решения о наказании и строится GUI.
                return
            }
            "ban" -> {
                plugin.logger.warning("[DaynAC] БАН: ${player.name} (средний скор $average, IP-бан: ${if (banWithIp) "да" else "нет"})")
                // ban + IP берёт свой шаблон (пустой ключ наследует ban) и подставляет
                // {action} = ban-ip, чтобы текст мог отличить жёсткий бан от обычного.
                val message = renderPunishMessage(
                    if (banWithIp) punishMessageBanIp else punishMessageBan,
                    player,
                    average,
                    if (banWithIp) "ban-ip" else "ban"
                )
                try {
                    if (banWithIp) {
                        // banPlayerFull = бан профиля + бан по IP адресу (и кик)
                        player.banPlayerFull(message)
                    } else {
                        // ban() = бан только по профилю (UUID), без IP; с киком.
                        // Приведение к Date обязательно: в новых версиях API есть перегрузки
                        // с Instant/Duration, и голый null стал бы неоднозначным.
                        player.ban(message, null as java.util.Date?, message, true)
                    }
                } catch (e: LinkageError) {
                    // Метод Bukkit API изменён/удалён в этой версии сервера — бан консольной командой
                    plugin.logger.warning("[DaynAC] API бана недоступно (${e.javaClass.simpleName}) — использую команду консоли")
                    runConsoleCommand(if (banWithIp) "ban-ip" else "ban", player.name, punishReason)
                }
            }
            else -> { // "kick" и любое другое значение, прошедшее require
                plugin.logger.warning("[DaynAC] Кик: ${player.name} (средний скор $average)")
                val message = renderPunishMessage(punishMessageKick, player, average, "kick")
                try {
                    player.kickPlayer(message)
                } catch (e: LinkageError) {
                    plugin.logger.warning("[DaynAC] API кика недоступно (${e.javaClass.simpleName}) — использую команду консоли")
                    runConsoleCommand("kick", player.name, punishReason)
                }
            }
        }

        // Уведомление админам в чат о факте наказания: обычные оповещения могли
        // быть подавлены кулдауном прямо перед киком/баном, и без этой строки
        // наказание видно только в консоли. Консольный лог дублирующего текста
        // избегаем (log = false) — у веток выше он уже есть.
        sendAlert(
            "[DaynAC] §4" + player.name + when {
                punishAction == "kick" -> "§7: авто-кик — средний скор §e"
                punishAction == "ban" && banWithIp -> "§7: авто-бан (профиль + IP) — средний скор §e"
                else -> "§7: авто-бан — средний скор §e"
            } + percent + "%§7",
            log = false
        )

        // Кик/бан — игрока на сервере больше нет: убираем из GUI и tracking-карт
        predictions.remove(player.uniqueId)
        scoreHistory.remove(player.uniqueId)
        hitCounts.remove(player.uniqueId)
        lastHitTimes.remove(player.uniqueId)
        punishProgress.remove(player.uniqueId)
        carryOver.remove(player.uniqueId)
        alertStates.remove(player.uniqueId)
    }

    /** Запасной путь наказания: команда от имени консоли (причина — одной строкой). Только с главного потока. */
    private fun runConsoleCommand(command: String, target: String, reason: String) {
        val oneLine = reason.replace('\n', ' ').replace('\r', ' ').trim()
        Bukkit.dispatchCommand(Bukkit.getConsoleSender(), "$command $target $oneLine")
    }

    /**
     * Снимает с игрока флаги: активные предсказания стираются, поэтому урон
     * снова проходит целиком, а автонаказание не срабатывает, пока игрок не
     * наберёт окно ударов заново ([FeatureExtractor.extract] до этого возвращает
     * null — предсказаний нет).
     *
     * @param keepCarryOver поведение при сбросе по простою:
     *   - true (default, вызов из [DaynAC.resetIfIdle]): если последнее окно было
     *     подозрительным (средний скор ≥ reduceThreshold), его скор запоминается
     *     в [carryOver] и линейно затухает за [carryOverSeconds]. Читер не получает
     *     «чистый лист» после 30 секунд тишины — первый удар нового боя всё ещё
     *     может пройти со сниженным уроном.
     *   - false (ручная команда /daynac clearflags): админ явно решил снять
     *     подозрение, carry-over стирается безусловно — урон снова полный
     *     с первого удара.
     *
     * Журнал ([scoreHistory]) и счётчик ударов ([hitCounts]) СОЗНАТЕЛЬНО остаются:
     * это витрина админа в /daynac allinfo — игрок не исчезает из списка, а его
     * «Последние удары» и «Макс» переживают сброс. В карточке такой игрок
     * показан как «Флаги сняты после простоя».
     *
     * Кулдаун оповещений завершается [flushPendingAlert]: накопленный агрегат
     * подавленных ударов уходит админам итоговым сообщением (иначе он бы не
     * показался никогда — нового подозрительного удара может не быть), после
     * чего кулдаун снят целиком и первый подозрительный удар нового боя
     * оповещает мгновенно.
     *
     * Окно ударов (HitTracker) и тайминги (SnapshotTracker) чистит вызывающая
     * сторона — движок их не знает.
     */
    fun resetFlags(playerId: UUID, keepCarryOver: Boolean = true) {
        // Бой окончен — последняя возможность показать подавленную кулдауном
        // активность: без выпуска она пропала бы, если новый подозрительный
        // удар не наступит. Агрегат уходит админам итоговым сообщением,
        // кулдаун снимается целиком.
        flushPendingAlert(playerId)
        if (keepCarryOver) {
            // Читер не должен получать «чистый лист» после 30 секунд тишины: иначе первые
            // windowSize ударов каждого нового боя идут без вердикта и без снижения урона.
            // Запоминаем средний скор перед сбросом; затухает за carryOverSeconds.
            predictions[playerId]?.let { deque ->
                val snapshot = synchronized(deque) { deque.toList() }
                if (snapshot.isNotEmpty()) {
                    val average = snapshot.sumOf { it.probability } / snapshot.size
                    if (carryOverSeconds > 0 && average >= reduceThreshold) {
                        carryOver[playerId] = CarryOver(average, clock())
                    } else {
                        carryOver.remove(playerId) // последнее окно было чистым — подозрение снято
                    }
                }
            }
        } else {
            // Ручной сброс (/daynac clearflags): чистим carry-over безусловно, даже
            // если активных предсказаний уже нет — запись могла остаться от прошлого
            // сброса по простою и продолжила бы снижать урон в новом бою.
            carryOver.remove(playerId)
        }
        predictions.remove(playerId)
        lastHitTimes.remove(playerId)
        punishProgress.remove(playerId) // окно ударов начинается заново — серия подтверждений тоже
    }

    /**
     * Выход игрока: предсказания СОХРАНЯЕМ (GUI показывает и оффлайн-игроков,
     * как в daynac.md — статус «● Не в сети»).
     */
    fun removePlayer(player: Player) {
        // Ничего не чистим: история предсказаний нужна GUI и после выхода
    }

    fun shutdown() {
        inferencePool.shutdownNow()
    }

    companion object {
        /**
         * Шаблоны по умолчанию — копия секции punish-message из config.yml. Раньше
         * этот текст собирался жёстко в [punishPlayer], и правка конфига ничего
         * не меняла: теперь они вшиты только как умолчание, а не как единственный
         * источник текста.
         */
        private val DEFAULT_PUNISH_MESSAGE_KICK = listOf(
            "&c[DaynAC] &f{reason}",
            "&7Средняя вероятность: &f{score}",
            "&7Если это ошибка — сообщите администрации."
        )

        private val DEFAULT_PUNISH_MESSAGE_BAN = listOf(
            "&cВы были забанены на этом сервере.",
            "&7Если это ошибка — обратитесь к администрации."
        )

        /** Кодовые символы, переводимые из &-формата: цвета, форматирование, сброс. */
        private const val LEGACY_CODE_CHARS = "0123456789abcdefklmnor"

        /** Hex-цвет &#RRGGBB — регистр цифр и букв не важен. */
        private val HEX_COLOR = Regex("[0-9a-fA-F]{6}")
    }
}