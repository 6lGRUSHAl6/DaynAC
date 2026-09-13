package zov.grusha.daynAC.detection

import org.bukkit.Bukkit
import org.bukkit.entity.Player
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
 * (327 → 128 → 64 → 32 → 1) не блокирует основной тик сервера. Решения о наказании
 * и модификации урона принимаются только на главном потоке через scheduler.
 *
 * Защита от ложных срабатываний — двойная:
 *  1. Множитель урона считается по ПРЕДЫДУЩИМ предсказаниям: первый удар всегда
 *     пропускается полностью (нет данных — нет наказания).
 *  2. Автонаказание требует и свежего предсказания ≥ punishThreshold, и среднего
 *     по последним [MAX_RECENT_PREDICTIONS] ударам ≥ punishThreshold.
 */
class DetectionEngine(
    private val plugin: org.bukkit.plugin.java.JavaPlugin,
    private val network: NeuralNetwork
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

    /** Автонаказание (кик) включено? false = режим наблюдения: только оповещения и снижение урона. */
    val punishEnabled: Boolean

    /** Тип наказания: "none" | "kick" | "ban". */
    val punishAction: String

    /** Причина наказания для кика/бана. */
    val punishReason: String

    init {
        val config = plugin.config
        flagThreshold = config.getDouble("detection.flag-threshold", 0.65)
        reduceThreshold = config.getDouble("detection.reduce-threshold", 0.70)
        cancelThreshold = config.getDouble("detection.cancel-threshold", 0.90)
        punishThreshold = config.getDouble("detection.punish-threshold", 0.95)
        maxRecentPredictions = config.getInt("detection.max-recent-predictions", 10)
        punishEnabled = config.getBoolean("detection.punish-enabled", true)
        punishAction = config.getString("detection.punish-action", "kick")?.lowercase() ?: "kick"
        punishReason = config.getString("detection.punish-reason", "KillAura (обнаружено нейросетью DaynAC)")!!

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
        require(maxRecentPredictions >= 1) { "detection.max-recent-predictions должен быть ≥ 1, получено $maxRecentPredictions" }
    }

    private data class PlayerVerdict(
        val probability: Double,
        val timestamp: Long
    )

    /** Записи о неактивных игроках старше этого возраста удаляются (TTL против монотонного роста карт). */
    private val trackedTtlMs = 60L * 60L * 1000L // 1 час

    private val inferencePool: ExecutorService = Executors.newFixedThreadPool(2) { runnable ->
        Thread(runnable, "DaynAC-Inference").apply { isDaemon = true }
    }

    /** Последние предсказания по игрокам (для усреднения и GUI/ver). */
    private val predictions = ConcurrentHashMap<UUID, ArrayDeque<PlayerVerdict>>()

    /** Сколько ударов игрок сделал (для /daynac ver и GUI — видно, почему нет предсказаний). */
    val hitCounts = java.util.concurrent.ConcurrentHashMap<UUID, Int>()

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
    }

    /** Сколько ударов у игрока и хватает ли их для анализа (окно 16). */
    fun getHitCount(playerId: UUID): Int = hitCounts[playerId] ?: 0

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
        // Счётчик ударов мал сам по себе; удаляем только записи оффлайн-игроков,
        // которых уже нет и в предсказаниях (иначе GUI потеряет «прогресс окна»).
        hitCounts.keys.retainAll { uuid ->
            predictions.containsKey(uuid) || Bukkit.getPlayer(uuid) != null
        }
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
        val deque = predictions.computeIfAbsent(attacker.uniqueId) { ArrayDeque() }
        synchronized(deque) {
            if (deque.size >= maxRecentPredictions) deque.removeFirst()
            deque.addLast(PlayerVerdict(probability, System.currentTimeMillis()))
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
        val p = getAveragePrediction(attacker) ?: return 1.0

        return when {
            p >= cancelThreshold -> 0.0
            p >= reduceThreshold -> {
                val t = (p - reduceThreshold) / (cancelThreshold - reduceThreshold)
                1.0 / (1.2 + t * 0.3)
            }
            else -> 1.0
        }
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

    /** Последние предсказания игрока (новые в конце), для GUI. */
    fun getRecentPredictions(playerId: UUID): List<Double> {
        val deque = predictions[playerId] ?: return emptyList()
        return synchronized(deque) { deque.map { it.probability } }
    }

    /** Максимальный скор за историю наблюдений игрока. */
    fun getMaxPrediction(playerId: UUID): Double? =
        getRecentPredictions(playerId).maxOrNull()

    /** Все UUID, по которым есть хоть одно предсказание (для GUI). */
    fun getTrackedPlayerIds(): Set<UUID> = predictions.keys.toSet()

    /**
     * Проверка порогов на главном потоке: оповещение админов при подозрении,
     * автонаказание при устойчиво высоком скоре.
     * @return true, если игрока нужно наказать (кик/бан — решает вызывающий код)
     */
    fun handlePrediction(attacker: Player, probability: Double): Boolean {
        val average = getAveragePrediction(attacker) ?: probability

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
            }
            Bukkit.getOnlinePlayers().filter { it.isOp }.forEach { it.sendMessage(message) }
            plugin.logger.warning(message.replace("§[0-9a-f]".toRegex(), ""))
        }

        // Автонаказание: отключено конфигом (режим наблюдения) или
        // мгновенный скор и среднее оба за порогом
        return punishEnabled && probability >= punishThreshold && average >= punishThreshold
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
     *  ban  — бан + кик (встроенный бан-лист сервера)
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
            }
            "ban" -> {
                plugin.logger.warning("[DaynAC] БАН: ${player.name} (средний скор $average)")
                player.banPlayerFull(message)  // бан + кик, встроенный бан-лист
            }
            else -> { // "kick" и любое другое значение, прошедшее require
                plugin.logger.warning("[DaynAC] Кик: ${player.name} (средний скор $average)")
                player.kickPlayer(message)
            }
        }
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
