package zov.grusha.daynAC.ml

import org.bukkit.configuration.file.YamlConfiguration
import java.io.File
import java.security.MessageDigest

fun main(args: Array<String>) {
    val dataDir = File(args.getOrNull(0) ?: "run/plugins/DaynAC")
    val epochs = args.getOrNull(1)?.toIntOrNull()?.takeIf { it > 0 } ?: 1000
    val patienceOverride = args.getOrNull(2)?.toIntOrNull() ?: -1
    val outFile = args.getOrNull(3)?.let(::File)

    val cfg = YamlConfiguration.loadConfiguration(File(dataDir, "config.yml"))
    fun number(path: String, default: Double): Double {
        val raw = cfg.get(path) ?: return default
        return when (raw) {
            is Number -> raw.toDouble()
            is String -> raw.trim().toDoubleOrNull() ?: error("$path: не число: $raw")
            else -> error("$path: не число: $raw")
        }
    }
    val lr = number("training.learning-rate", 0.001)
    val l2 = number("training.l2-lambda", 1e-4)
    val dropout = number("training.dropout-rate", 0.15)
    val patience = if (patienceOverride >= 0) patienceOverride else number("training.patience", 15.0).toInt()
    val splitBy = (cfg.getString("training.split-by", "session") ?: "session").lowercase()
    val hidden = when (val raw = cfg.get("hidden-layers")) {
        is Number -> intArrayOf(raw.toInt())
        is List<*> -> raw.map { (it as Number).toInt() }.toIntArray()
        else -> intArrayOf(64, 32, 16)
    }
    val inputSize = 16 * FeatureExtractor.PER_HIT_FEATURES + FeatureExtractor.AGGREGATE_FEATURES

    val datasetReadStart = System.nanoTime()
    val dataset = DatasetManager(dataDir, windowSize = 16).readCompactDataset()
    val datasetReadMs = (System.nanoTime() - datasetReadStart) / 1e6

    val nn = NeuralNetwork(inputSize = inputSize, hiddenSizes = hidden, seed = 42L)
    val groups = dataset.groupsBy(splitBy)

    val trainStart = System.nanoTime()
    val epochEndNs = ArrayList<Long>(epochs)
    val result = nn.train(
        dataset = dataset,
        groups = groups,
        learningRate = lr,
        maxEpochs = epochs,
        patience = patience,
        l2Lambda = l2,
        dropoutRate = dropout,
        onProgress = { _, _, _, _ -> epochEndNs.add(System.nanoTime()) }
    )
    val trainMs = (System.nanoTime() - trainStart) / 1e6

    // onProgress вызывается один раз на эпоху при valCheckInterval=1 (maxEpochs ≤ 50).
    // Дельта между вызовами = время эпохи вместе с val-проверкой.
    val epochMs = ArrayList<Double>(epochEndNs.size)
    var prev = trainStart
    for (t in epochEndNs) {
        epochMs.add((t - prev) / 1e6)
        prev = t
    }
    fun avg(list: List<Double>) = if (list.isEmpty()) 0.0 else list.sum() / list.size
    val first10 = avg(epochMs.take(10))
    val last10 = avg(epochMs.takeLast(10))

    val modelFile = File(System.getProperty("java.io.tmpdir"), "daynac-bench-${ProcessHandle.current().pid()}.dat")
    val saveStart = System.nanoTime()
    nn.saveTo(modelFile)
    val saveMs = (System.nanoTime() - saveStart) / 1e6
    val modelHash = MessageDigest.getInstance("SHA-256")
        .digest(modelFile.readBytes())
        .joinToString("") { "%02x".format(it) }
    modelFile.delete()

    val thresholds = doubleArrayOf(0.5, 0.65, 0.7, 0.9, 0.95)
    val confusions = nn.evaluateThresholds(thresholds)!!.map {
        "${it.threshold}:TP=${it.truePositives},FP=${it.falsePositives},FN=${it.falseNegatives},TN=${it.trueNegatives}"
    }

    fun d(v: Double) = String.format(java.util.Locale.US, "%.12f", v)
    val report = buildString {
        appendLine("{")
        appendLine("  \"dataset\": {\"legit\": ${dataset.legitCount}, \"cheat\": ${dataset.cheatCount}, \"total\": ${dataset.sampleCount}},")
        appendLine("  \"config\": {\"lr\": $lr, \"l2\": $l2, \"dropout\": $dropout, \"patience\": $patience, \"split\": \"$splitBy\", \"hidden\": ${hidden.contentToString()}, \"maxEpochs\": $epochs},")
        appendLine("  \"timing_ms\": {\"dataset_read\": ${d(datasetReadMs)}, \"train\": ${d(trainMs)}, \"save\": ${d(saveMs)}, \"per_epoch\": ${d(trainMs / result.epochsRun)}, \"first10\": ${d(first10)}, \"last10\": ${d(last10)}},")
        appendLine("  \"result\": {\"epochsRun\": ${result.epochsRun}, \"bestEpoch\": ${result.bestEpoch}, \"earlyStopped\": ${result.earlyStopped},")
        appendLine("    \"trainLoss\": ${d(result.trainLoss)}, \"valLoss\": ${d(result.valLoss)}, \"testLoss\": ${d(result.testLoss ?: Double.NaN)},")
        appendLine("    \"trainAcc\": ${d(result.trainAccuracy)}, \"valAcc\": ${d(result.valAccuracy)}, \"testAcc\": ${d(result.testAccuracy ?: Double.NaN)}},")
        appendLine("  \"epoch_ms\": [${epochMs.joinToString(",") { String.format(java.util.Locale.US, "%.1f", it) }}],")
        appendLine("  \"split\": \"${result.split}\",")
        appendLine("  \"thresholds\": \"${confusions.joinToString("; ")}\",")
        appendLine("  \"model_sha256\": \"$modelHash\"")
        appendLine("}")
    }
    println(report)
    outFile?.let {
        it.parentFile?.mkdirs()
        it.appendText(report + "\n")
    }
}
