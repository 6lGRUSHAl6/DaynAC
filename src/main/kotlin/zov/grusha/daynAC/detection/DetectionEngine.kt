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
    private val windowSize: Int
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

    init {
        val config = plugin.config
        flagThreshold = config.getDouble("detection.flag-threshold", 0.65)
        reduceThreshold = config.getDouble("detection.reduce-threshold", 0.70)
        cancelThreshold = config.getDouble("detection.cancel-threshold", 0.90)
        punishThreshold = config.getDouble("detection.punish-threshold", 0.95)
        maxRecentPredictions = config.getInt("detection.max-recent-predictions", 10)
        punishConfirmWindows = config.getInt("detection.punish-confirm-windows", 2)
        inactivityResetSeconds = config.getInt("detection.inactivity-reset-seconds", 30)
        carryOverSeconds = config.getInt("detection.carry-over-seconds", 600)
        punishEnabled = config.getBoolean("detection.punish-enabled", true)
        punishAction = config.getString("detection.punish-action", "kick")?.lowercase() ?: "kick"
        punishReason = config.getString("detection.punish-reason", "KillAura (обнаружено нейросетью DaynAC)")!!
        banWithIp = config.getBoolean("detection.ban-with-ip", false)

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

    private val inferencePool: ExecutorService = Executors.newFixedThreadPool(2) { runnable ->
        Thread(runnable, "DaynAC-Inference").apply { isDaemon = true }
    }

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

    /** Сколько ударов игрок сделал (для /daynac ver и GUI — видно, почему нет предсказаний). */
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
        lastHitTimes[player.uniqueId] = System.currentTimeMillis()
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
        return System.currentTimeMillis() - lastHit > inactivityResetSeconds * 1000L
    }

    /**
     * Снимает с игрока флаги после простоя: активные предсказания стираются,
     * поэтому урон снова проходит целиком, а автонаказание не срабатывает, пока
     * игрок не наберёт окно ударов заново ([FeatureExtractor.extract] до этого
     * возвращает null — предсказаний нет).
     *
     * Журнал ([scoreHistory]) и счётчик ударов ([hitCounts]) СОЗНАТЕЛЬНО остаются:
     * это витрина админа в /daynac allinfo — игрок не исчезает из списка, а его
     * «Последние удары» и «Макс» переживают сброс. В карточке такой игрок
     * показан как «Флаги сняты после простоя».
     *
     * Окно ударов (HitTracker) и тайминги (SnapshotTracker) чистит вызывающая
     * сторона — движок их не знает.
     */
    fun resetFlags(playerId: UUID) {
        // Читер не должен получать «чистый лист» после 30 секунд тишины: иначе первые
        // windowSize ударов каждого нового боя идут без вердикта и без снижения урона.
        // Запоминаем средний скор перед сбросом; затухает за carryOverSeconds.
        predictions[playerId]?.let { deque ->
            val snapshot = synchronized(deque) { deque.toList() }
            if (snapshot.isNotEmpty()) {
                val average = snapshot.sumOf { it.probability } / snapshot.size
                if (carryOverSeconds > 0 && average >= reduceThreshold) {
                    carryOver[playerId] = CarryOver(average, System.currentTimeMillis())
                } else {
                    carryOver.remove(playerId) // последнее окно было чистым — подозрение снято
                }
            }
        }
        predictions.remove(playerId)
        lastHitTimes.remove(playerId)
        punishProgress.remove(playerId) // окно ударов начинается заново — серия подтверждений тоже
    }

    /**
     * Удаляет записи о неактивных игроках старше TTL. Предсказания и счётчики
     * ударов сохраняются после выхода игрока специально (GUI показывает и
     * оффлайн-игроков), но без ограничения срока они копятся бесконечно на
     * долгоживущем сервере. Вызывается периодически из onEnable — независимо
     * от того, открывает ли кто-то GUI.
     */
    fun pruneStaleTracking() {
        val now = System.currentTimeMillis()
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
        val verdict = PlayerVerdict(probability, System.currentTimeMillis())

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
        val elapsed = System.currentTimeMillis() - carried.timeMillis
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
        val progress = updatePunishProgress(attacker.uniqueId, probability)

        // Мгновенное оповещение на каждый необычный удар — без таймеров и кулдаунов:
        // скор этого удара превысил flag-threshold => пишем сразу
        if (probability >= flagThreshold) {
            // Сводка по последним 4 ударам — динамика вокруг необычного удара
            val recent4 = getRecentPredictions(attacker.uniqueId).takeLast(4)
            val scoresLine = recent4.joinToString(" ") { score ->
                scoreColor(score) + String.format(Locale.US, "%.2f", score)
            }

            val message = buildString {
                append("[DaynAC] §c")
                append(attacker.name)
                append("§7: необычный удар §e")
                append(String.format(Locale.US, "%.3f", probability))
                append("§7 > порога §f")
                append(flagThreshold)
                append("§7 — последние удары: ")
                append(scoresLine)
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
            }
            Bukkit.getOnlinePlayers().filter { it.hasPermission(Permissions.ALERTS) }.forEach { it.sendMessage(message) }
            plugin.logger.warning(message.replace("§[0-9a-f]".toRegex(), ""))
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
     * Наказание согласно detection.punish-action из конфига:
     *  none — только запись в лог (полное бездействие)
     *  kick — кик с сообщением
     *  ban  — бан; при detection.ban-with-ip=true дополнительно бан по IP
     *         (banPlayerFull), иначе только профиль (UUID)
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
        val message = buildString {
            append("§c[DaynAC] $punishReason\n")
            append("§7Средняя вероятность: §f$percent%\n")
            append("§7Если это ошибка — сообщите администрации.")
        }

        when (punishAction) {
            "none" -> {
                plugin.logger.warning("[DaynAC] Наказание отключено (punish-action: none): ${player.name}, средний скор $average")
                // Игрок остался на сервере: историю предсказаний сохраняем —
                // по ней считается среднее для решения о наказании и строится GUI.
                return
            }
            "ban" -> {
                plugin.logger.warning("[DaynAC] БАН: ${player.name} (средний скор $average, IP-бан: ${if (banWithIp) "да" else "нет"})")
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
                try {
                    player.kickPlayer(message)
                } catch (e: LinkageError) {
                    plugin.logger.warning("[DaynAC] API кика недоступно (${e.javaClass.simpleName}) — использую команду консоли")
                    runConsoleCommand("kick", player.name, punishReason)
                }
            }
        }

        // Кик/бан — игрока на сервере больше нет: убираем из GUI и tracking-карт
        predictions.remove(player.uniqueId)
        scoreHistory.remove(player.uniqueId)
        hitCounts.remove(player.uniqueId)
        lastHitTimes.remove(player.uniqueId)
        punishProgress.remove(player.uniqueId)
        carryOver.remove(player.uniqueId)
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
     * Окно ударов (HitTracker) и тайминги (SnapshotTracker) чистит вызывающая
     * сторона — движок их не знает.
     */
    fun resetFlags(playerId: UUID, keepCarryOver: Boolean = true) {
        if (keepCarryOver) {
            // Читер не должен получать «чистый лист» после 30 секунд тишины: иначе первые
            // windowSize ударов каждого нового боя идут без вердикта и без снижения урона.
            // Запоминаем средний скор перед сбросом; затухает за carryOverSeconds.
            predictions[playerId]?.let { deque ->
                val snapshot = synchronized(deque) { deque.toList() }
                if (snapshot.isNotEmpty()) {
                    val average = snapshot.sumOf { it.probability } / snapshot.size
                    if (carryOverSeconds > 0 && average >= reduceThreshold) {
                        carryOver[playerId] = CarryOver(average, System.currentTimeMillis())
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
}