package zov.grusha.daynAC

import org.bukkit.Bukkit
import org.bukkit.command.Command
import org.bukkit.command.CommandSender
import org.bukkit.entity.Player
import org.bukkit.event.EventHandler
import org.bukkit.event.Listener
import org.bukkit.event.entity.EntityDamageByEntityEvent
import org.bukkit.event.entity.EntityDamageEvent
import org.bukkit.event.player.PlayerQuitEvent
import org.bukkit.plugin.java.JavaPlugin
import zov.grusha.daynAC.tracking.SnapshotTracker
import java.util.Locale
import java.util.UUID
import zov.grusha.daynAC.tracking.HitTracker
import zov.grusha.daynAC.ml.DatasetManager
import zov.grusha.daynAC.ml.FeatureExtractor
import zov.grusha.daynAC.ml.NeuralNetwork
import zov.grusha.daynAC.detection.DetectionEngine
import zov.grusha.daynAC.gui.AllInfoGUI
import java.io.File
import kotlin.concurrent.thread

class DaynAC : JavaPlugin(), Listener {

    companion object {
        /**
         * Единственный источник правды для размера окна ударов. От него зависят:
         * размер истории в HitTracker, длина вектора FeatureExtractor
         * и заголовок векторных CSV в DatasetManager. Менять только здесь.
         */
        const val WINDOW_SIZE = 16
    }

    private val hitCounter = mutableMapOf<Pair<UUID, UUID>, Int>()
    private val snapshotTracker = SnapshotTracker(maxSize = 20)
    private val hitTracker = HitTracker(maxSize = WINDOW_SIZE)
    private val featureExtractor = FeatureExtractor(windowSize = WINDOW_SIZE)

    private val datasetManager = DatasetManager(dataFolder, windowSize = WINDOW_SIZE)

    private val inputSize = WINDOW_SIZE * FeatureExtractor.PER_HIT_FEATURES + FeatureExtractor.AGGREGATE_FEATURES
    private val neuralNetwork = NeuralNetwork(inputSize = inputSize)
    private val detectionEngine = DetectionEngine(this, neuralNetwork)
    private val allInfoGUI = AllInfoGUI(this, detectionEngine)
    private val modelFile = File(dataFolder, "model.dat")

    /** Обучение идёт в фоне — блокируем повторный запуск /daynac train. */
    @Volatile
    private var training = false

    /** Подробное логгирование ударов (чат опам + консоль). Управляется /daynac debug. */
    @Volatile
    private var hitLogging: Boolean = false

    // ИЗМЕНЕНИЕ: Теперь храним статус записи для каждого игрока отдельно (по UUID)
    private val recordingPlayers = mutableMapOf<UUID, String>()

    override fun onEnable() {
        logger.info("Enabling plugin...")
        saveDefaultConfig()
        hitLogging = config.getBoolean("hit-logging", false)
        server.pluginManager.registerEvents(this, this)
        server.pluginManager.registerEvents(allInfoGUI, this)
        Bukkit.getScheduler().runTaskTimer(this, Runnable {
            for (player in Bukkit.getOnlinePlayers()) {
                snapshotTracker.recordSnapshot(player)
            }
        }, 0L, 1L)

        // Загрузка обученной модели, если она совместима с текущей структурой признаков
        if (neuralNetwork.isCompatibleWith(modelFile)) {
            try {
                neuralNetwork.loadFrom(modelFile)
                detectionEngine.enableDetection()
                logger.info("Модель загружена из model.dat — детекция активна.")
            } catch (e: Exception) {
                logger.warning("Не удалось загрузить модель: ${e.message} — детекция отключена.")
            }
        } else {
            logger.info("Модель не найдена или несовместима (вход $inputSize) — детекция отключена до /daynac train.")
        }
    }

    override fun onDisable() {
        logger.info("Disabling plugin...")
        detectionEngine.shutdown()
    }

    override fun onCommand(sender: CommandSender, command: Command, label: String, args: Array<out String>): Boolean {
        if (command.name.equals("daynac", ignoreCase = true)) {
            when {
                args.isNotEmpty() && args[0] == "ver" -> {
                    if (args.size < 2) {
                        // Без аргументов — общий статус
                        sender.sendMessage(
                            "daynac v${description.version} | модель: " +
                                (if (detectionEngine.detectionEnabled) "активна" else "не обучена") +
                                " | вход: $inputSize признаков"
                        )
                        return true
                    }
                    val target = Bukkit.getPlayerExact(args[1])
                    if (target == null) {
                        sender.sendMessage("Игрок ${args[1]} не найден или не в сети.")
                        return true
                    }
                    if (!detectionEngine.detectionEnabled) {
                        sender.sendMessage("Модель не обучена — сначала /daynac train.")
                        return true
                    }
                    val last = detectionEngine.getLastPrediction(target)
                    val avg = detectionEngine.getAveragePrediction(target)
                    val hits = detectionEngine.getHitCount(target.uniqueId)
                    sender.sendMessage(
                        "[DaynAC] ${target.name}: последний скор = " +
                            (last?.let { String.format(Locale.US, "%.3f", it) } ?: "нет данных") +
                            ", средний = " + (avg?.let { String.format(Locale.US, "%.3f", it) } ?: "нет данных") +
                            ", ударов = $hits" + if (hits < DaynAC.WINDOW_SIZE) " (нужно ${DaynAC.WINDOW_SIZE} для анализа)" else ""
                    )
                }
                args.isNotEmpty() && args[0] == "train" -> {
                    if (training) {
                        sender.sendMessage("Обучение уже идёт, дождитесь завершения.")
                        return true
                    }

                    val (features, labels, parsedCounts) = datasetManager.readVectorSamples()
                    val (legitCount, cheatCount) = parsedCounts

                    if (features.size < 10) {
                        sender.sendMessage(
                            "Недостаточно данных: $legitCount легит / $cheatCount чит векторов " +
                                "(нужно ≥ 10). Соберите через /daynac record."
                        )
                        return true
                    }
                    if (legitCount == 0 || cheatCount == 0) {
                        sender.sendMessage(
                            "Датасет однобок: $legitCount легит / $cheatCount чит. " +
                                "Нужны оба класса — соберите и legit, и cheat сэмплы."
                        )
                        return true
                    }

                    training = true
                    sender.sendMessage("Обучение запущено: $legitCount легит / $cheatCount чит векторов...")

                    thread(name = "DaynAC-Training", isDaemon = true) {
                        // Отчёты отправляем на главный поток — Bukkit API из async-потока нельзя
                        fun report(message: String) {
                            Bukkit.getScheduler().runTask(this@DaynAC, Runnable { sender.sendMessage(message) })
                        }

                        try {
                            val startTime = System.currentTimeMillis()
                            val result = neuralNetwork.train(
                                samples = features,
                                labels = labels,
                                learningRate = 0.001,
                                maxEpochs = 1000,
                                patience = 15
                            )

                            neuralNetwork.saveTo(modelFile)
                            detectionEngine.enableDetection()

                            val elapsed = (System.currentTimeMillis() - startTime) / 1000.0
                            report(buildString {
                                fun pct(v: Double) = String.format(Locale.US, "%.1f%%", v * 100)
                                fun num(v: Double) = String.format(Locale.US, "%.4f", v)
                                appendLine("§a[DaynAC] Обучение завершено за ${String.format(Locale.US, "%.1f", elapsed)}с")
                                appendLine("§7Эпох: §f${result.epochsRun} §7(лучшая: ${result.bestEpoch}" +
                                    if (result.earlyStopped) ", early stop)" else ")")
                                appendLine("§7Train loss/acc: §f${num(result.trainLoss)} / ${pct(result.trainAccuracy)}")
                                appendLine("§7Val loss/acc:   §f${num(result.valLoss)} / ${pct(result.valAccuracy)}")
                                appendLine(
                                    if (result.overfitGap > 0.05) "§eПереобучение: разрыв train/val ${pct(result.overfitGap)} — соберите больше данных"
                                    else "§aРазрыв train/val: ${pct(result.overfitGap)} — в норме"
                                )
                                append("§aМодель сохранена, детекция активна.")
                            })
                        } catch (e: Exception) {
                            logger.severe("Обучение упало: ${e.message}")
                            report("§c[DaynAC] Ошибка обучения: ${e.message}")
                        } finally {
                            training = false
                        }
                    }
                }
                args.isNotEmpty() && args[0] == "allinfo" -> {
                    val admin = sender as? Player
                    if (admin == null) {
                        sender.sendMessage("Команда доступна только игрокам.")
                        return true
                    }
                    if (!detectionEngine.detectionEnabled) {
                        sender.sendMessage("Модель не обучена — сначала /daynac train. GUI покажет пустой список.")
                    }
                    allInfoGUI.open(admin)
                }
                args.isNotEmpty() && args[0] == "debug" -> {
                    // Переключение подробного логгирования ударов (сохраняется в config.yml)
                    hitLogging = !hitLogging
                    config.set("hit-logging", hitLogging)
                    saveConfig()
                    sender.sendMessage("[DaynAC] Логгирование ударов: ${if (hitLogging) "§aвключено" else "§cвыключено"}")
                }
                args.isNotEmpty() && args[0] == "info" -> {
                    val (legitCount, cheatCount) = datasetManager.getVectorDatasetStats()
                    sender.sendMessage(
                        buildString {
                            appendLine("[DaynAC] Датасет: $legitCount легит / $cheatCount чит векторов")
                            appendLine("[DaynAC] Модель: " + (if (detectionEngine.detectionEnabled) "активна" else "не обучена") +
                                ", вход: $inputSize признаков (окно $WINDOW_SIZE ударов)")
                            append("[DaynAC] Пороги: flag ${detectionEngine.flagThreshold} / reduce ${detectionEngine.reduceThreshold}" +
                                " / cancel ${detectionEngine.cancelThreshold} / punish ${detectionEngine.punishThreshold}" +
                                " / окно усреднения ${detectionEngine.maxRecentPredictions}")
                            appendLine()
                            append("[DaynAC] Наказание: " + when {
                                !detectionEngine.punishEnabled -> "§eВЫКЛЮЧЕНО (режим наблюдения)"
                                detectionEngine.punishAction == "none" -> "§7ничего (только лог)"
                                else -> "§c${detectionEngine.punishAction}"
                            })
                        }
                    )
                }
                args.isNotEmpty() && args[0] == "reload" -> {
                    // Перезагрузка config.yml. Пороги зашиты в DetectionEngine на момент
                    // конструирования, поэтому после reload требуется перезапуск сервера —
                    // предупреждаем об этом честно.
                    reloadConfig()
                    hitLogging = config.getBoolean("hit-logging", hitLogging)
                    sender.sendMessage("§a[DaynAC] config.yml перезагружен (hit-logging: $hitLogging).")
                    sender.sendMessage("§eПороги детекции читаются при старте плагина — для их применения перезапустите сервер.")
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
                    sender.sendMessage("Неизвестная команда. Доступно: /daynac <ver|record|train|info|debug|allinfo|reload>")
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
                1 -> { // Подсказка для первого аргумента
                    return listOf("ver", "record", "train", "info", "debug", "allinfo", "reload")
                        .filter { it.startsWith(args[0], ignoreCase = true) }
                        .toMutableList()
                }
                2 -> { // Подсказка для второго аргумента (ник игрока)
                    if (args[0].equals("ver", ignoreCase = true) || args[0].equals("record", ignoreCase = true)) {
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

    // ignoreCancelled: удары, отменённые другими плагинами (регионы, God-режим),
    // не должны попадать в историю, датасет и запускать inference.
    // Фильтр по cause: в 1.9+ sweep-атака мечом генерирует отдельные события
    // для всех задетых игроков с почти нулевым hitTimeDelta и aimAngle на
    // случайную жертву — тайминговые признаки выглядят механическими.
    @EventHandler(ignoreCancelled = true)
    fun onEntityDamagePlayer(event: EntityDamageByEntityEvent) {
        if (event.cause != EntityDamageEvent.DamageCause.ENTITY_ATTACK) return
        val attacker = event.damager as? Player ?: return
        val victim = event.entity as? Player ?: return

        val hitData = snapshotTracker.buildHitData(attacker, victim)
        hitTracker.addHit(attacker, hitData)
        detectionEngine.recordHit(attacker)

        // ИЗМЕНЕНИЕ: Проверяем, записываем ли мы именно этого атакующего
        recordingPlayers[attacker.uniqueId]?.let { label ->
            datasetManager.writeSample(label, hitData)

            // Вектор признаков пишем только на каждый WINDOW_SIZE-й удар
            // (stride = windowSize). При записи на каждый удар соседние окна
            // совпадают на 15/16 признаков, и случайный train/val split в
            // обучении кладёт почти-дубликаты в обе выборки — val-метрики
            // фиктивны, early stopping ломается.
            if (detectionEngine.getHitCount(attacker.uniqueId) % WINDOW_SIZE == 0) {
                featureExtractor.extract(hitTracker.getHistory(attacker))?.let { vector ->
                    datasetManager.writeVectorSample(label, vector)
                }
            }
        }

        // Детекция: асинхронный inference. Записывающего игрока не проверяем —
        // иммунитет на время сбора данных, чтобы не наказывать за те же удары.
        if (detectionEngine.detectionEnabled && !recordingPlayers.containsKey(attacker.uniqueId)) {
            // Множитель по ПРЕДЫДУЩИМ предсказаниям — первый удар всегда полный
            val multiplier = detectionEngine.getDamageMultiplier(attacker)
            if (multiplier <= 0.0) {
                event.isCancelled = true
            } else if (multiplier < 1.0) {
                event.damage = event.damage * multiplier
            }

            featureExtractor.extract(hitTracker.getHistory(attacker))?.let { vector ->
                detectionEngine.analyzeHitAsync(attacker, vector) { player, probability ->
                    if (detectionEngine.handlePrediction(player, probability)) {
                        detectionEngine.punishPlayer(player)
                    }
                }
            }
        }

        val pairKey = Pair(attacker.uniqueId, victim.uniqueId)
        val currentHits = hitCounter.compute(pairKey) { _, count -> (count ?: 0) + 1 }

        // Подробное логгирование каждого удара — включается/выключается через /daynac debug
        if (hitLogging) {
            val message = "[LOG] ${attacker.name} hit ${victim.name} (count=$currentHits) $hitData"
            Bukkit.getOnlinePlayers().filter { it.isOp }.forEach { it.sendMessage(message) }
            logger.info(message)
        }
    }

    @EventHandler
    fun onPlayerQuit(event: PlayerQuitEvent) {
        snapshotTracker.removePlayer(event.player)
        hitTracker.removePlayer(event.player)
        detectionEngine.removePlayer(event.player)
        recordingPlayers.remove(event.player.uniqueId)
    }

    private fun formatDouble(value: Double?, format: String = "%.4f"): String {
        return value?.let { String.format(Locale.US, format, it) } ?: "N/A"
    }
}