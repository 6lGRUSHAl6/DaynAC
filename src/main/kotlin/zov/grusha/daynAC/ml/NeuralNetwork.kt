package zov.grusha.daynAC.ml

import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.util.Random
import java.util.concurrent.atomic.AtomicReference
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * MLP-классификатор «читер / легит» на чистом Kotlin.
 *
 * Скрытые слои — ReLU, выход — Sigmoid (вероятность чита). Обучение: BCE, backprop,
 * Adam по одному сэмплу, L2, inverted dropout, z-score входов, early stopping по validation.
 *
 * Потокобезопасность: [predict] не берёт блокировок и читает неизменяемый снапшот весов;
 * [train] и [loadFrom] сериализуются [mutationLock] и публикуют результат одним атомарным set().
 */
class NeuralNetwork(
    val inputSize: Int,
    hiddenSizes: IntArray = intArrayOf(8, 16),
    private val seed: Long = 42L
) {

    val layerSizes: IntArray = intArrayOf(inputSize, *hiddenSizes, 1)

    /**
     * Состояние сети. Объект, опубликованный через [snapshot], менять нельзя (copy-on-write):
     * train мутирует локальную копию и публикует её целиком.
     */
    private class Weights(
        val weights: Array<Array<DoubleArray>>,
        val biases: Array<DoubleArray>,
        val featureMean: DoubleArray,
        val featureStd: DoubleArray,
        val mWeights: Array<Array<DoubleArray>>,
        val vWeights: Array<Array<DoubleArray>>,
        val mBiases: Array<DoubleArray>,
        val vBiases: Array<DoubleArray>
    ) {
        fun deepCopy() = Weights(
            weights.deepCopy(), biases.deepCopy(), featureMean.copyOf(), featureStd.copyOf(),
            mWeights.deepCopy(), vWeights.deepCopy(), mBiases.deepCopy(), vBiases.deepCopy()
        )

        fun copyFrom(other: Weights) {
            weights.copyFrom(other.weights)
            biases.copyFrom(other.biases)
            other.featureMean.copyInto(featureMean)
            other.featureStd.copyInto(featureStd)
            mWeights.copyFrom(other.mWeights)
            vWeights.copyFrom(other.vWeights)
            mBiases.copyFrom(other.mBiases)
            vBiases.copyFrom(other.vBiases)
        }
    }

    /** Буферы активаций, масок dropout и дельт: один набор на всё обучение, без аллокаций на сэмпл. */
    private class StepContext(
        layerSizes: IntArray,
        val learningRate: Double,
        val l2Lambda: Double,
        val dropoutRate: Double
    ) {
        val activations = Array(layerSizes.size - 1) { DoubleArray(layerSizes[it + 1]) }
        val masks = Array(layerSizes.size - 1) { DoubleArray(layerSizes[it + 1]) }
        val deltas = Array(layerSizes.size - 1) { DoubleArray(layerSizes[it + 1]) }
        var biasCorrection1 = 1.0
        var biasCorrection2 = 1.0
    }

    /**
     * Нормализованный test-сет последнего train(): модель на нём не училась и эпоха по нему
     * не выбиралась. В model.dat не сохраняется — после рестарта eval недоступен до train().
     */
    private class HoldoutSet(
        val normalizedFeatures: List<DoubleArray>,
        val labels: DoubleArray
    )

    @Volatile
    private var lastTestSet: HoldoutSet? = null

    private fun emptyWeights(): Weights {
        fun matrices() = Array(layerSizes.size - 1) { l -> Array(layerSizes[l + 1]) { DoubleArray(layerSizes[l]) } }
        fun vectors() = Array(layerSizes.size - 1) { l -> DoubleArray(layerSizes[l + 1]) }
        return Weights(
            weights = matrices(),
            biases = vectors(),
            featureMean = DoubleArray(inputSize),
            featureStd = DoubleArray(inputSize) { 1.0 },
            mWeights = matrices(),
            vWeights = matrices(),
            mBiases = vectors(),
            vBiases = vectors()
        )
    }

    private val snapshot = AtomicReference(emptyWeights())

    /**
     * Сериализует train и loadFrom: иначе train, стартовавший до loadFrom, затёр бы
     * загруженную модель своим set(). Читатели (predict, saveTo) лок не берут.
     */
    private val mutationLock = ReentrantLock()

    companion object {
        private const val EPS = 1e-15
        private const val CLAMP = 5.0
        private const val SIGMOID_CLAMP = 500.0
        private const val MIN_STD = 1e-8
        private const val BETA1 = 0.9
        private const val BETA2 = 0.999
        private const val ADAM_EPS = 1e-8
        private const val MAGIC = 0x44414E41
        private const val FORMAT_VERSION = 3   // v2: + метрики обучения; v3: + версия признаков

        /** Доли validation (early stopping) и test (финальная оценка) на класс; остальное — train. */
        private const val VAL_FRACTION = 0.2
        private const val TEST_FRACTION = 0.2

        /** Минимум сэмплов на класс: при 5 получается 3 train / 1 val / 1 test. */
        const val MIN_SAMPLES_PER_CLASS = 5

        /** Минимум групп на класс для разбиения по группам (по одной в train, val и test). */
        const val MIN_GROUPS_PER_CLASS = 3
    }

    private fun relu(x: Double): Double = if (x > 0.0) x else 0.0

    private fun sigmoid(x: Double): Double =
        1.0 / (1.0 + exp(-x.coerceIn(-SIGMOID_CLAMP, SIGMOID_CLAMP)))

    /** Вероятность чита в (0, 1) по «сырому» вектору признаков; z-score применяется внутри. */
    fun predict(rawFeatures: DoubleArray): Double {
        require(rawFeatures.size == inputSize) {
            "Ожидался вектор из $inputSize признаков, получен ${rawFeatures.size}"
        }
        val state = snapshot.get()
        return forwardOnly(state, normalize(rawFeatures, state))
    }

    private fun normalize(features: DoubleArray, state: Weights): DoubleArray {
        val out = DoubleArray(features.size)
        for (i in features.indices) {
            out[i] = (features[i] - state.featureMean[i]) / state.featureStd[i]
        }
        return out
    }

    private fun newLayerBuffers() = Array(layerSizes.size - 1) { DoubleArray(layerSizes[it + 1]) }

    private fun forwardLayer(state: Weights, l: Int, input: DoubleArray, output: DoubleArray, isLast: Boolean) {
        val weights = state.weights[l]
        val biases = state.biases[l]
        for (j in output.indices) {
            val wj = weights[j]
            var z = biases[j]
            for (k in input.indices) {
                z += wj[k] * input[k]
            }
            output[j] = if (isLast) sigmoid(z) else relu(z)
        }
    }

    private fun forwardOnly(
        state: Weights,
        normalizedFeatures: DoubleArray,
        buffers: Array<DoubleArray> = newLayerBuffers()
    ): Double {
        var activation = normalizedFeatures
        val lastLayer = state.weights.size - 1
        for (l in 0..lastLayer) {
            forwardLayer(state, l, activation, buffers[l], l == lastLayer)
            activation = buffers[l]
        }
        return activation[0]
    }

    /**
     * Как разбит датасет. `*Grouped` = true: группы (сессии или игроки) класса целиком попали
     * в один сет, утечки между train и test нет. false: разбиение случайное по окнам,
     * окна одного игрока могут оказаться и в train, и в test — метрики оптимистичны.
     */
    data class SplitSummary(
        val legitGrouped: Boolean,
        val cheatGrouped: Boolean,
        val legitGroups: Int,
        val cheatGroups: Int,
        val trainSize: Int,
        val valSize: Int,
        val testSize: Int
    ) {
        val fullyGrouped: Boolean get() = legitGrouped && cheatGrouped
    }

    /** Результат обучения. Поля test* и [split] равны null, если результат загружен из model.dat. */
    data class TrainResult(
        val epochsRun: Int,
        val bestEpoch: Int,
        val earlyStopped: Boolean,
        val trainLoss: Double,
        val valLoss: Double,
        val trainAccuracy: Double,
        val valAccuracy: Double,
        val testLoss: Double? = null,
        val testAccuracy: Double? = null,
        val split: SplitSummary? = null
    ) {
        /** Разрыв train/val accuracy — грубая метрика переобучения (0.05+ уже подозрительно). */
        val overfitGap: Double get() = trainAccuracy - valAccuracy
    }

    /** Метрики последнего обучения; устанавливаются в train() и loadFrom(). */
    @Volatile
    var lastTrainResult: TrainResult? = null
        private set

    /**
     * Стратифицированный split 60/20/20 (train/val/test), z-score по train, обучение по одному
     * сэмплу с early stopping по val и восстановлением лучших весов. Test-сет в обучении
     * и выборе эпохи не участвует.
     *
     * @param samples векторы признаков длиной [inputSize]
     * @param labels 0.0 = легит, 1.0 = чит
     * @param learningRate шаг Adam (оптимально 0.001–0.003)
     * @param maxEpochs максимальное число эпох
     * @param patience число проверок val loss без улучшения до остановки
     * @param l2Lambda коэффициент L2 weight decay (0 — отключить)
     * @param dropoutRate доля отключаемых нейронов скрытых слоёв
     * @param groups группа каждого сэмпла (сессия или игрок). Если на класс приходится
     *               ≥ [MIN_GROUPS_PER_CLASS] групп, разбиение идёт по группам
     * @param onProgress колбэк прогресса (эпоха, trainLoss, valLoss, valAcc); valLoss и valAcc
     *                   равны null на эпохах без проверки
     */
    fun train(
        samples: List<DoubleArray>,
        labels: List<Double>,
        learningRate: Double = 0.002,
        maxEpochs: Int = 2000,
        patience: Int = 40,
        l2Lambda: Double = 1e-4,
        dropoutRate: Double = 0.15,
        groups: List<String?>? = null,
        onProgress: ((epoch: Int, trainLoss: Double, valLoss: Double?, valAcc: Double?) -> Unit)? = null
    ): TrainResult {
        require(samples.size == labels.size) { "samples и labels разной длины" }
        require(samples.size >= 10) { "Датасет слишком мал: ${samples.size} сэмплов (нужно ≥ 10)" }
        require(samples.all { it.size == inputSize }) { "Все сэмплы должны быть длиной $inputSize" }
        require(groups == null || groups.size == samples.size) { "groups и samples разной длины" }
        val legitTotal = labels.count { it == 0.0 }
        val cheatTotal = labels.count { it == 1.0 }
        require(legitTotal >= MIN_SAMPLES_PER_CLASS && cheatTotal >= MIN_SAMPLES_PER_CLASS) {
            "Нужно ≥ $MIN_SAMPLES_PER_CLASS сэмплов каждого класса (legit: $legitTotal, cheat: $cheatTotal)"
        }

        mutationLock.withLock {
            lastTestSet = null

            val local = snapshot.get().deepCopy()
            val split = splitIndices(labels, groups, Random(seed))

            initWeights(local)
            computeNormStats(local, split.train.map { samples[it] })

            fun featuresOf(idx: List<Int>) = idx.map { normalize(samples[it], local) }
            fun labelsOf(idx: List<Int>) = DoubleArray(idx.size) { labels[idx[it]] }

            val trainFeatures = featuresOf(split.train)
            val valFeatures = featuresOf(split.validation)
            val testFeatures = featuresOf(split.test)
            val trainLabels = labelsOf(split.train)
            val valLabels = labelsOf(split.validation)
            val testLabels = labelsOf(split.test)

            val context = StepContext(layerSizes, learningRate, l2Lambda, dropoutRate)
            val best = local.deepCopy()
            val order = IntArray(trainFeatures.size)
            val valCheckInterval = max(1, min(5, maxEpochs / 50))
            var bestValLoss = Double.POSITIVE_INFINITY
            var bestEpoch = 0
            var staleCount = 0
            var earlyStopped = false
            var epoch = 0

            while (epoch < maxEpochs) {
                val epochRng = Random(seed + epoch + 1)
                for (i in order.indices) order[i] = i
                order.shuffleInPlace(epochRng)

                val t = (epoch + 1).toDouble()
                context.biasCorrection1 = 1.0 - BETA1.pow(t)
                context.biasCorrection2 = 1.0 - BETA2.pow(t)

                var epochLoss = 0.0
                for (i in order) {
                    val label = trainLabels[i]
                    val prediction = trainStep(local, context, trainFeatures[i], label, epochRng)
                    epochLoss += bceLoss(prediction, label)
                }
                val trainLoss = epochLoss / order.size

                if (epoch % valCheckInterval == 0) {
                    val valResult = evaluateNormalized(local, valFeatures, valLabels)
                    onProgress?.invoke(epoch, trainLoss, valResult.avgLoss, valResult.accuracy)

                    if (valResult.avgLoss < bestValLoss) {
                        bestValLoss = valResult.avgLoss
                        bestEpoch = epoch
                        staleCount = 0
                        best.copyFrom(local)
                    } else {
                        staleCount++
                        if (patience > 0 && staleCount >= patience) {
                            earlyStopped = true
                            break
                        }
                    }
                } else {
                    onProgress?.invoke(epoch, trainLoss, null, null)
                }

                epoch++
            }

            local.copyFrom(best)

            val finalTrain = evaluateNormalized(local, trainFeatures, trainLabels)
            val finalVal = evaluateNormalized(local, valFeatures, valLabels)
            val finalTest = evaluateNormalized(local, testFeatures, testLabels)

            lastTestSet = HoldoutSet(testFeatures, testLabels)
            snapshot.set(local)

            val result = TrainResult(
                epochsRun = epoch,
                bestEpoch = bestEpoch,
                earlyStopped = earlyStopped,
                trainLoss = finalTrain.avgLoss,
                valLoss = finalVal.avgLoss,
                trainAccuracy = finalTrain.accuracy,
                valAccuracy = finalVal.accuracy,
                testLoss = finalTest.avgLoss,
                testAccuracy = finalTest.accuracy,
                split = split.summary
            )
            lastTrainResult = result
            return result
        }
    }

    /**
     * Матрица ошибок при фиксированном пороге. Для античита критичнее precision/recall:
     * FPR — доля легитных окон, ошибочно помеченных как чит; FNR — доля пропущенных читов.
     */
    data class ThresholdMetrics(
        val threshold: Double,
        val truePositives: Int,
        val falsePositives: Int,
        val trueNegatives: Int,
        val falseNegatives: Int,
        val totalLegit: Int,
        val totalCheat: Int
    ) {
        val precision: Double get() =
            if (truePositives + falsePositives == 0) 0.0
            else truePositives.toDouble() / (truePositives + falsePositives)

        val recall: Double get() =
            if (truePositives + falseNegatives == 0) 0.0
            else truePositives.toDouble() / (truePositives + falseNegatives)

        val f1: Double get() =
            if (precision + recall == 0.0) 0.0
            else 2.0 * precision * recall / (precision + recall)

        val falsePositiveRate: Double get() =
            if (totalLegit == 0) 0.0
            else falsePositives.toDouble() / totalLegit

        val falseNegativeRate: Double get() =
            if (totalCheat == 0) 0.0
            else falseNegatives.toDouble() / totalCheat

        /** 95% интервал Уилсона для FPR: на малой выборке точечная оценка вводит в заблуждение. */
        val falsePositiveRateInterval: Pair<Double, Double> get() =
            Stats.wilsonInterval(falsePositives, totalLegit)

        /** 95% интервал Уилсона для recall. */
        val recallInterval: Pair<Double, Double> get() =
            Stats.wilsonInterval(truePositives, totalCheat)
    }

    /**
     * Матрицы ошибок на test-сете последнего train() для каждого порога из [thresholds].
     * @return null, если в этом сеансе ещё не было train()
     */
    fun evaluateThresholds(thresholds: DoubleArray): List<ThresholdMetrics>? {
        val testSet = lastTestSet ?: return null
        val state = snapshot.get()

        val buffers = newLayerBuffers()
        val probabilities = testSet.normalizedFeatures.map { forwardOnly(state, it, buffers) }

        val totalLegit = testSet.labels.count { it < 0.5 }
        val totalCheat = testSet.labels.count { it >= 0.5 }

        return thresholds.map { threshold ->
            var tp = 0; var fp = 0; var tn = 0; var fn = 0
            for (i in probabilities.indices) {
                val predictedCheat = probabilities[i] >= threshold
                val actualCheat = testSet.labels[i] >= 0.5
                when {
                    predictedCheat && actualCheat -> tp++
                    predictedCheat && !actualCheat -> fp++
                    !predictedCheat && !actualCheat -> tn++
                    else -> fn++
                }
            }
            ThresholdMetrics(threshold, tp, fp, tn, fn, totalLegit, totalCheat)
        }
    }

    private class IndexSplit(
        val train: List<Int>,
        val validation: List<Int>,
        val test: List<Int>,
        val summary: SplitSummary
    )

    /**
     * Делит индексы на train/val/test (60/20/20) отдельно по каждому классу.
     *
     * Если у класса ≥ [MIN_GROUPS_PER_CLASS] групп, делит целыми группами: они перемешиваются
     * и раздаются в test, затем в val, остальные идут в train (в каждом сете минимум одна группа).
     * Строки без группы считаются отдельными группами по одному окну. Иначе — случайное
     * разбиение окон, что отражено в [SplitSummary].
     */
    private fun splitIndices(labels: List<Double>, groups: List<String?>?, rng: Random): IndexSplit {
        val train = ArrayList<Int>()
        val validation = ArrayList<Int>()
        val test = ArrayList<Int>()
        val grouped = BooleanArray(2)
        val groupCounts = IntArray(2)

        for ((c, label) in listOf(0.0, 1.0).withIndex()) {
            val classIdx = labels.indices.filter { labels[it] == label }
            val byGroup = LinkedHashMap<String, MutableList<Int>>()
            for (i in classIdx) {
                val g = groups?.get(i) ?: continue
                byGroup.getOrPut(g) { ArrayList() }.add(i)
            }
            groupCounts[c] = byGroup.size

            val testTarget = max(1, (classIdx.size * TEST_FRACTION).toInt())
            val valTarget = max(1, (classIdx.size * VAL_FRACTION).toInt())

            if (byGroup.size >= MIN_GROUPS_PER_CLASS) {
                grouped[c] = true
                val units = ArrayList<List<Int>>(byGroup.values)
                for (i in classIdx) if (groups?.get(i) == null) units.add(listOf(i))
                units.shuffle(rng)

                val classTest = ArrayList<Int>()
                val classVal = ArrayList<Int>()
                var next = 0
                // test: минимум одна группа, затем добираем, пока остаются ≥ 2 группы (val + train)
                do { classTest.addAll(units[next]); next++ } while (classTest.size < testTarget && units.size - next > 2)
                // val: минимум одна группа, в train остаётся ≥ 1
                do { classVal.addAll(units[next]); next++ } while (classVal.size < valTarget && units.size - next > 1)
                test.addAll(classTest)
                validation.addAll(classVal)
                for (j in next until units.size) train.addAll(units[j])
            } else {
                val shuffled = classIdx.shuffled(rng)
                test.addAll(shuffled.take(testTarget))
                validation.addAll(shuffled.drop(testTarget).take(valTarget))
                train.addAll(shuffled.drop(testTarget + valTarget))
            }
        }

        val summary = SplitSummary(
            legitGrouped = grouped[0], cheatGrouped = grouped[1],
            legitGroups = groupCounts[0], cheatGroups = groupCounts[1],
            trainSize = train.size, valSize = validation.size, testSize = test.size
        )
        return IndexSplit(train, validation, test, summary)
    }

    /** He-инициализация весов (оптимальна для ReLU), bias и моменты Adam — нули. */
    private fun initWeights(state: Weights) {
        val rng = Random(seed)
        for (l in state.weights.indices) {
            val scale = sqrt(2.0 / layerSizes[l])
            state.biases[l].fill(0.0)
            state.mBiases[l].fill(0.0)
            state.vBiases[l].fill(0.0)
            for (j in state.weights[l].indices) {
                val row = state.weights[l][j]
                for (k in row.indices) row[k] = rng.nextGaussian() * scale
                state.mWeights[l][j].fill(0.0)
                state.vWeights[l][j].fill(0.0)
            }
        }
    }

    /**
     * Forward + backward + шаг Adam по одному сэмплу. Дельта выхода для BCE+Sigmoid — (ŷ − y).
     * Градиент веса: clip(δ·a + l2·w).
     * @return предсказание до обновления (для подсчёта loss)
     */
    private fun trainStep(state: Weights, ctx: StepContext, features: DoubleArray, label: Double, rng: Random): Double {
        val layerCount = state.weights.size
        val lastLayer = layerCount - 1
        val activations = ctx.activations
        val deltas = ctx.deltas
        val dropoutRate = ctx.dropoutRate
        val useDropout = dropoutRate > 0.0
        val keepProbability = 1.0 - dropoutRate

        for (l in 0 until layerCount) {
            val input = if (l == 0) features else activations[l - 1]
            val output = activations[l]
            forwardLayer(state, l, input, output, l == lastLayer)

            if (l != lastLayer && useDropout) {
                val mask = ctx.masks[l]
                for (j in output.indices) {
                    if (rng.nextDouble() < dropoutRate) {
                        output[j] = 0.0
                        mask[j] = 0.0
                    } else {
                        output[j] /= keepProbability
                        mask[j] = 1.0
                    }
                }
            }
        }

        val prediction = activations[lastLayer][0]
        deltas[lastLayer][0] = prediction - label

        for (l in lastLayer - 1 downTo 0) {
            val delta = deltas[l]
            val nextDelta = deltas[l + 1]
            val nextWeights = state.weights[l + 1]
            val activation = activations[l]
            val mask = if (useDropout) ctx.masks[l] else null

            delta.fill(0.0)
            for (n in nextDelta.indices) {
                val row = nextWeights[n]
                val dn = nextDelta[n]
                for (j in delta.indices) {
                    delta[j] += row[j] * dn
                }
            }
            for (j in delta.indices) {
                var d = if (activation[j] > 0.0) delta[j] else 0.0
                if (mask != null) d *= mask[j]
                delta[j] = d
            }
        }

        val learningRate = ctx.learningRate
        val l2Lambda = ctx.l2Lambda
        val bias1 = ctx.biasCorrection1
        val bias2 = ctx.biasCorrection2

        for (l in 0 until layerCount) {
            val prev = if (l == 0) features else activations[l - 1]
            val delta = deltas[l]
            val weights = state.weights[l]
            val mWeights = state.mWeights[l]
            val vWeights = state.vWeights[l]
            val biases = state.biases[l]
            val mBiases = state.mBiases[l]
            val vBiases = state.vBiases[l]

            for (j in weights.indices) {
                val dj = delta[j]
                val wj = weights[j]
                val mj = mWeights[j]
                val vj = vWeights[j]

                for (k in wj.indices) {
                    // min/max вместо coerceIn: без ветвлений JIT векторизует цикл
                    val grad = Math.min(CLAMP, Math.max(-CLAMP, dj * prev[k] + l2Lambda * wj[k]))
                    val m = BETA1 * mj[k] + (1.0 - BETA1) * grad
                    val v = BETA2 * vj[k] + (1.0 - BETA2) * grad * grad
                    mj[k] = m
                    vj[k] = v
                    wj[k] -= learningRate * (m / bias1) / (sqrt(v / bias2) + ADAM_EPS)
                }

                val mb = BETA1 * mBiases[j] + (1.0 - BETA1) * dj
                val vb = BETA2 * vBiases[j] + (1.0 - BETA2) * dj * dj
                mBiases[j] = mb
                vBiases[j] = vb
                biases[j] -= learningRate * (mb / bias1) / (sqrt(vb / bias2) + ADAM_EPS)
            }
        }

        return prediction
    }

    private fun bceLoss(prediction: Double, label: Double): Double {
        val p = prediction.coerceIn(EPS, 1.0 - EPS)
        return -(label * ln(p) + (1.0 - label) * ln(1.0 - p))
    }

    /** Средний BCE и accuracy (порог 0.5) на уже нормализованных сэмплах. */
    private fun evaluateNormalized(state: Weights, features: List<DoubleArray>, labels: DoubleArray): Evaluation {
        if (features.isEmpty()) return Evaluation(0.0, 0.0)
        val buffers = newLayerBuffers()
        var totalLoss = 0.0
        var correct = 0
        for (i in features.indices) {
            val prediction = forwardOnly(state, features[i], buffers)
            totalLoss += bceLoss(prediction, labels[i])
            val predictedClass = if (prediction >= 0.5) 1.0 else 0.0
            if (predictedClass == labels[i]) correct++
        }
        return Evaluation(totalLoss / features.size, correct.toDouble() / features.size)
    }

    private data class Evaluation(val avgLoss: Double, val accuracy: Double)

    /** Z-score статистики по train-выборке; константный признак получает std = 1.0. */
    private fun computeNormStats(state: Weights, trainFeatures: List<DoubleArray>) {
        val n = trainFeatures.size
        for (i in 0 until inputSize) {
            var mean = 0.0
            for (sample in trainFeatures) mean += sample[i]
            mean /= n

            var variance = 0.0
            for (sample in trainFeatures) {
                val d = sample[i] - mean
                variance += d * d
            }
            variance /= n

            val std = sqrt(variance)
            state.featureMean[i] = mean
            state.featureStd[i] = if (std < MIN_STD) 1.0 else std
        }
    }

    /** Проверяет, что модель в файле совместима с текущим [inputSize]. */
    fun isCompatibleWith(file: File): Boolean {
        if (!file.exists()) return false
        return try {
            DataInputStream(file.inputStream().buffered()).use { input ->
                input.readInt() == MAGIC &&
                        input.readInt() == FORMAT_VERSION &&
                        input.readInt() == FeatureExtractor.FEATURE_VERSION &&
                        input.readInt() == inputSize &&
                        input.readInt() == layerSizes.size
            }
        } catch (_: Exception) {
            false
        }
    }

    fun saveTo(file: File) {
        val state = snapshot.get()
        // Пишем во временный файл и переименовываем: обрыв посреди записи
        // не должен оставлять model.dat с валидным заголовком и обрезанными весами.
        val tmpFile = File(file.parentFile, file.name + ".tmp")
        try {
            DataOutputStream(tmpFile.outputStream().buffered()).use { output ->
                output.writeInt(MAGIC)
                output.writeInt(FORMAT_VERSION)
                output.writeInt(FeatureExtractor.FEATURE_VERSION)
                output.writeInt(inputSize)
                output.writeInt(layerSizes.size)
                for (size in layerSizes) output.writeInt(size)

                for (layer in state.weights) {
                    for (neuron in layer) {
                        for (w in neuron) output.writeDouble(w)
                    }
                }
                for (layer in state.biases) {
                    for (b in layer) output.writeDouble(b)
                }
                for (m in state.featureMean) output.writeDouble(m)
                for (s in state.featureStd) output.writeDouble(s)

                val tr = lastTrainResult
                if (tr != null) {
                    output.writeBoolean(true)
                    output.writeInt(tr.epochsRun)
                    output.writeInt(tr.bestEpoch)
                    output.writeBoolean(tr.earlyStopped)
                    output.writeDouble(tr.trainLoss)
                    output.writeDouble(tr.valLoss)
                    output.writeDouble(tr.trainAccuracy)
                    output.writeDouble(tr.valAccuracy)
                } else {
                    output.writeBoolean(false)
                }
            }
            if (file.exists()) {
                Files.move(tmpFile.toPath(), file.toPath(), StandardCopyOption.REPLACE_EXISTING)
            } else {
                Files.move(tmpFile.toPath(), file.toPath(), StandardCopyOption.ATOMIC_MOVE)
            }
        } catch (e: Exception) {
            tmpFile.delete()
            throw e
        }
    }

    fun loadFrom(file: File) {
        mutationLock.withLock {
            DataInputStream(file.inputStream().buffered()).use { input ->
                check(input.readInt() == MAGIC) { "Файл модели повреждён (неверный маркер)" }
                check(input.readInt() == FORMAT_VERSION) { "Неподдерживаемая версия формата модели" }
                val modelFeatureVersion = input.readInt()
                check(modelFeatureVersion == FeatureExtractor.FEATURE_VERSION) {
                    "Модель обучена на версии признаков $modelFeatureVersion, текущая — ${FeatureExtractor.FEATURE_VERSION}. Переобучите: /daynac train"
                }
                check(input.readInt() == inputSize) { "Размер входа модели не совпадает с текущим ($inputSize)" }
                val layerCount = input.readInt()
                check(layerCount == layerSizes.size) { "Структура слоёв модели не совпадает" }
                for (l in layerSizes.indices) {
                    check(input.readInt() == layerSizes[l]) { "Размер слоя $l не совпадает" }
                }

                val loaded = emptyWeights()
                for (layer in loaded.weights) {
                    for (neuron in layer) {
                        for (k in neuron.indices) neuron[k] = input.readDouble()
                    }
                }
                for (layer in loaded.biases) {
                    for (j in layer.indices) layer[j] = input.readDouble()
                }
                for (i in loaded.featureMean.indices) loaded.featureMean[i] = input.readDouble()
                for (i in loaded.featureStd.indices) loaded.featureStd[i] = input.readDouble()

                // Блок метрик может отсутствовать, если файл закончился на весах.
                lastTrainResult = if (input.available() > 0 && input.readBoolean()) {
                    TrainResult(
                        epochsRun = input.readInt(),
                        bestEpoch = input.readInt(),
                        earlyStopped = input.readBoolean(),
                        trainLoss = input.readDouble(),
                        valLoss = input.readDouble(),
                        trainAccuracy = input.readDouble(),
                        valAccuracy = input.readDouble()
                    )
                } else null

                lastTestSet = null
                snapshot.set(loaded)
            }
        }
    }
}

private fun Array<DoubleArray>.deepCopy() = Array(size) { this[it].copyOf() }

private fun Array<Array<DoubleArray>>.deepCopy() = Array(size) { this[it].deepCopy() }

private fun Array<DoubleArray>.copyFrom(src: Array<DoubleArray>) {
    for (i in indices) src[i].copyInto(this[i])
}

private fun Array<Array<DoubleArray>>.copyFrom(src: Array<Array<DoubleArray>>) {
    for (i in indices) this[i].copyFrom(src[i])
}

/** Тот же алгоритм, что у [java.util.Collections.shuffle], — порядок для одного seed не меняется. */
private fun IntArray.shuffleInPlace(random: Random) {
    for (i in size downTo 2) {
        val j = random.nextInt(i)
        val tmp = this[i - 1]
        this[i - 1] = this[j]
        this[j] = tmp
    }
}