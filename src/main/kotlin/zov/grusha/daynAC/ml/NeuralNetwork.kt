package zov.grusha.daynAC.ml

import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.util.Random
import java.util.concurrent.atomic.AtomicReference
import java.util.concurrent.locks.ReentrantLock
import java.io.EOFException
import kotlin.concurrent.withLock
import java.nio.file.AtomicMoveNotSupportedException
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
 *
 * Память: обучение принимает [TrainingDataset] с признаками в одном плоском массиве и
 * нормализует каждый сэмпл в небольшой буфер прямо перед шагом SGD — вторая копия
 * всего датасета не создаётся. Test-сет хранится не как нормализованные вектора, а как
 * два отсортированных массива вероятностей (по классу): это позволяет считать любые
 * пороги бинарным поиском и не раздувать model.dat пропорционально датасету.
 */
class NeuralNetwork(
    val inputSize: Int,
    hiddenSizes: IntArray = intArrayOf(64, 32, 16),
    private val seed: Long = 42L
) {

    /**
     * Размеры всех слоёв, включая вход и выход. Может меняться на лету через
     * [setArchitecture] — тогда состояние сети сбрасывается, а model.dat с другой
     * архитектурой считается несовместимым (см. isCompatibleWith).
     */
    @Volatile
    private var layerSizes: IntArray = intArrayOf(inputSize, *hiddenSizes, 1)

    /** Скрытые слои без входа и выхода — для /daynac info и сообщений reload. */
    val hiddenLayerSizes: IntArray
        get() = layerSizes.copyOfRange(1, layerSizes.size - 1)

    /**
     * Перестраивает сеть под новые скрытые слои. Возвращает false, если архитектура
     * уже такая (ничего не изменилось); true — если перестроена.
     *
     * Перестройка сбрасывает веса в ноль и очищает lastTestSet/lastTrainResult:
     * после смены архитектуры старая модель в любом случае несовместима, и сеть
     * нужно обучать заново или загружать совместимую model.dat. Под mutationLock —
     * сериализуется с train/loadFrom, и вызывающий код после true обязан
     * перезагрузить или обучить модель до включения детекции.
     */
    fun setArchitecture(newHiddenSizes: IntArray): Boolean {
        require(newHiddenSizes.isNotEmpty()) { "Нужен хотя бы один скрытый слой" }
        require(newHiddenSizes.all { it >= 1 }) { "Размер скрытого слоя должен быть ≥ 1, получено ${newHiddenSizes.contentToString()}" }
        mutationLock.withLock {
            val newSizes = intArrayOf(inputSize, *newHiddenSizes, 1)
            if (newSizes.contentEquals(layerSizes)) return false
            layerSizes = newSizes
            // Свежая архитектура → нулевые веса; predict даст мусор, поэтому
            // детекцию отключает вызывающий код.
            snapshot.set(emptyWeights())
            lastTestSet = null
            lastTrainResult = null
            return true
        }
    }

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
        val dropoutRate: Double,
        val fastMath: Boolean = false
    ) {
        val activations = Array(layerSizes.size - 1) { DoubleArray(layerSizes[it + 1]) }
        val masks = Array(layerSizes.size - 1) { DoubleArray(layerSizes[it + 1]) }
        val deltas = Array(layerSizes.size - 1) { DoubleArray(layerSizes[it + 1]) }
        var biasCorrection1 = 1.0
        var biasCorrection2 = 1.0
    }

    /**
     * Та же последовательность, что у java.util.Random, но seed хранится в
     * обычном Long. В JDK next(bits) идёт через AtomicLong с CAS;
     * в однопоточном обучении это чистая трата времени (десятки миллионов CAS
     * за эпоху на dropout-маски). nextDouble()/nextGaussian()/nextInt() унаследованы
     * и вызывают наш next(bits), поэтому распределение и порядок чисел не меняются.
     */
    internal class FastRandom(seed: Long) : Random(0L) {
        private var fastSeed: Long = (seed xor MULTIPLIER) and MASK

        override fun next(bits: Int): Int {
            fastSeed = (fastSeed * MULTIPLIER + ADDEND) and MASK
            return (fastSeed ushr (48 - bits)).toInt()
        }

        private companion object {
            const val MULTIPLIER = 0x5DEECE66DL
            const val ADDEND = 0xBL
            const val MASK = 0x0000FFFFFFFFFFFFL
        }
    }

    /**
     * Test-сет последнего train(): модель на нём не училась и эпоха по нему не выбиралась.
     *
     * Вместо нормализованных векторов (343 double на окно, т.е. ~2.7 КБ на сэмпл в
     * model.dat) храним только предсказанные вероятности — по одному double на окно,
     * раздельно для legit и cheat. Массивы отсортированы, поэтому долю выше любого
     * порога можно получить бинарным поиском, а /daynac eval работает и после
     * рестарта сервера (см. saveTo/loadFrom).
     */
    private class HoldoutSet(
        val legitProbabilities: DoubleArray,
        val cheatProbabilities: DoubleArray
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
     * Рабочие буферы [predict]: нормализованный вход и активации слоёв. Размеры
     * привязаны к конкретному снапшоту весов: при смене архитектуры через
     * [setArchitecture] буфер пересоздаётся под новый снапшот (см. [scratchFor]).
     * Свои на каждый поток: predict вызывается без блокировок из разных потоков
     * и не реентерабелен, так что общий буфер был бы гонкой данных.
     */
    private class Scratch(state: Weights, inputSize: Int) {
        val input = DoubleArray(inputSize)
        val layers = Array(state.weights.size) { DoubleArray(state.biases[it].size) }

        /** Годен ли буфер для этого снапшота (размеры слоёв совпадают). */
        fun matches(state: Weights, inputSize: Int): Boolean =
            input.size == inputSize &&
                    layers.size == state.weights.size &&
                    layers.indices.all { layers[it].size == state.biases[it].size }
    }

    private val scratchRef = ThreadLocal.withInitial { arrayOfNulls<Scratch>(1) }

    /** Буфер текущего потока, согласованный с размерами [state]. */
    private fun scratchFor(state: Weights): Scratch {
        val holder = scratchRef.get()
        val existing = holder[0]
        if (existing != null && existing.matches(state, inputSize)) return existing
        val fresh = Scratch(state, inputSize)
        holder[0] = fresh
        return fresh
    }

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

        /**
         * Опциональный быстрый путь шага Adam. Включается только явным
         * -Ddaynac.training.fast-math=true (для бенчмарка), по умолчанию выключен:
         * результат отличается от оригинала в последних битах.
         */
        private val FAST_MATH = System.getProperty("daynac.training.fast-math", "false")
            .equals("true", ignoreCase = true)
        private const val MAGIC = 0x44414E41
        private const val FORMAT_VERSION = 3   // v2: + метрики обучения; v3: + версия признаков.
        // Блок test-сета добавлен ниже метрик как необязательный хвост (версию не меняли,
        // чтобы старые model.dat оставались совместимы и не требовали переобучения).
        // Новый блок пишет отрицательный count как маркер «компактный формат»: внутри два
        // массива вероятностей вместо сырых нормализованных векторов. Читатель понимает
        // оба варианта, поэтому старые model.dat грузятся как раньше.
        private const val COMPACT_TEST_BLOCK = -1

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
        val buf = scratchFor(state)
        normalizeInto(rawFeatures, state, buf.input)
        return forwardOnly(state, buf.input, buf.layers)
    }

    private fun normalizeInto(features: DoubleArray, state: Weights, out: DoubleArray) {
        for (i in features.indices) {
            out[i] = (features[i] - state.featureMean[i]) / state.featureStd[i]
        }
    }

    /** Нормализация сэмпла `index` из плоского датасета в переиспользуемый буфер. */
    private fun normalizeAt(features: DoubleArray, index: Int, state: Weights, out: DoubleArray) {
        val base = index * inputSize
        for (i in 0 until inputSize) {
            out[i] = (features[base + i] - state.featureMean[i]) / state.featureStd[i]
        }
    }

    private fun newLayerBuffers(state: Weights) = Array(state.weights.size) { DoubleArray(state.biases[it].size) }

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
        buffers: Array<DoubleArray>
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

    /** Результат обучения. Поля test* и [split] могут быть null, если модель загружена
     *  из старого model.dat без блока test-сета (см. loadFrom). */
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
     * Совместимый API для тестов и маленьких датасетов: разворачивает список векторов в
     * плоский массив и делегирует в [trainInternal]. Большой датасет (/daynac train)
     * должен идти через перегрузку [train] с [TrainingDataset] — без лишних копий.
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
        maxEpochs: Int = 1000,
        patience: Int = 40,
        l2Lambda: Double = 1e-4,
        dropoutRate: Double = 0.15,
        groups: List<String?>? = null,
        onProgress: ((epoch: Int, trainLoss: Double, valLoss: Double?, valAcc: Double?) -> Unit)? = null
    ): TrainResult {
        require(samples.size == labels.size) { "samples и labels разной длины" }
        require(samples.all { it.size == inputSize }) { "Все сэмплы должны быть длиной $inputSize" }
        val flat = DoubleArray(samples.size * inputSize)
        for (i in samples.indices) samples[i].copyInto(flat, i * inputSize)
        return trainInternal(
            features = flat,
            labels = labels.toDoubleArray(),
            sampleCount = samples.size,
            groups = groups?.toTypedArray(),
            learningRate = learningRate,
            maxEpochs = maxEpochs,
            patience = patience,
            l2Lambda = l2Lambda,
            dropoutRate = dropoutRate,
            onProgress = onProgress
        )
    }

    /**
     * Основной путь обучения: признаки уже лежат в одном плоском массиве. [groups]
     * по умолчанию берёт grouping из датасета (сессии); фоновый поток обучения передаёт
     * `dataset.groupsBy(splitBy)`, прочитанный на главном потоке до старта.
     */
    fun train(
        dataset: TrainingDataset,
        groups: Array<String?>? = null,
        learningRate: Double = 0.002,
        maxEpochs: Int = 1000,
        patience: Int = 40,
        l2Lambda: Double = 1e-4,
        dropoutRate: Double = 0.15,
        onProgress: ((epoch: Int, trainLoss: Double, valLoss: Double?, valAcc: Double?) -> Unit)? = null
    ): TrainResult {
        require(dataset.featureCount == inputSize) {
            "Ожидалось $inputSize признаков на сэмпл, в датасете ${dataset.featureCount}"
        }
        return trainInternal(
            features = dataset.features,
            labels = dataset.labels,
            sampleCount = dataset.sampleCount,
            groups = groups ?: dataset.sessionIds,
            learningRate = learningRate,
            maxEpochs = maxEpochs,
            patience = patience,
            l2Lambda = l2Lambda,
            dropoutRate = dropoutRate,
            onProgress = onProgress
        )
    }

    /**
     * Стратифицированный split 60/20/20 (train/val/test), z-score по train, обучение по одному
     * сэмплу с early stopping по val и восстановлением лучших весов. Test-сет в обучении
     * и выборе эпохи не участвует.
     *
     * Нормализованные копии всех сэмплов НЕ создаются: [computeNormStats] делает один
     * последовательный проход по train-индексам, а каждый сэмпл нормализуется в общий
     * буфер непосредственно перед forward/backward.
     */
    private fun trainInternal(
        features: DoubleArray,
        labels: DoubleArray,
        sampleCount: Int,
        groups: Array<String?>?,
        learningRate: Double,
        maxEpochs: Int,
        patience: Int,
        l2Lambda: Double,
        dropoutRate: Double,
        onProgress: ((epoch: Int, trainLoss: Double, valLoss: Double?, valAcc: Double?) -> Unit)?
    ): TrainResult {
        require(features.size == sampleCount * inputSize) {
            "features должны содержать ровно sampleCount * inputSize чисел"
        }
        require(sampleCount >= 10) { "Датасет слишком мал: $sampleCount сэмплов (нужно ≥ 10)" }
        require(groups == null || groups.size == sampleCount) { "groups и samples разной длины" }
        val legitTotal = labels.count { it == 0.0 }
        val cheatTotal = labels.count { it == 1.0 }
        require(legitTotal >= MIN_SAMPLES_PER_CLASS && cheatTotal >= MIN_SAMPLES_PER_CLASS) {
            "Нужно ≥ $MIN_SAMPLES_PER_CLASS сэмплов каждого класса (legit: $legitTotal, cheat: $cheatTotal)"
        }

        mutationLock.withLock {
            lastTestSet = null

            val local = snapshot.get().deepCopy()
            val split = splitIndices(labels, groups, FastRandom(seed))

            initWeights(local)
            computeNormStats(local, features, split.train)

            val context = StepContext(layerSizes, learningRate, l2Lambda, dropoutRate, FAST_MATH)
            val best = local.deepCopy()
            val order = split.train.copyOf()
            val stepInput = DoubleArray(inputSize)
            val evalBuffers = newLayerBuffers(local)
            val valCheckInterval = max(1, min(5, maxEpochs / 50))
            var bestValLoss = Double.POSITIVE_INFINITY
            var bestEpoch = 0
            var staleCount = 0
            var earlyStopped = false
            var epoch = 0

            while (epoch < maxEpochs) {
                val epochRng = FastRandom(seed + epoch + 1)
                order.shuffleInPlace(epochRng)

                val t = (epoch + 1).toDouble()
                context.biasCorrection1 = 1.0 - BETA1.pow(t)
                context.biasCorrection2 = 1.0 - BETA2.pow(t)

                var epochLoss = 0.0
                for (idx in order) {
                    normalizeAt(features, idx, local, stepInput)
                    val prediction = trainStep(local, context, stepInput, labels[idx], epochRng)
                    epochLoss += bceLoss(prediction, labels[idx])
                }
                val trainLoss = epochLoss / order.size

                if (epoch % valCheckInterval == 0) {
                    val valResult = evaluateIndices(local, features, labels, split.validation, stepInput, evalBuffers)
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

            val finalTrain = evaluateIndices(local, features, labels, split.train, stepInput, evalBuffers)
            val finalVal = evaluateIndices(local, features, labels, split.validation, stepInput, evalBuffers)
            val (legitProbs, cheatProbs) = testProbabilities(local, features, labels, split.test, stepInput, evalBuffers)
            val finalTest = metricsOf(legitProbs, cheatProbs)

            lastTestSet = HoldoutSet(legitProbs, cheatProbs)
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

    /** Средний BCE и accuracy по подмножеству индексов плоского датасета. */
    private fun evaluateIndices(
        state: Weights,
        features: DoubleArray,
        labels: DoubleArray,
        indices: IntArray,
        input: DoubleArray,
        buffers: Array<DoubleArray>
    ): Evaluation {
        if (indices.isEmpty()) return Evaluation(0.0, 0.0)
        var totalLoss = 0.0
        var correct = 0
        for (idx in indices) {
            normalizeAt(features, idx, state, input)
            val prediction = forwardOnly(state, input, buffers)
            totalLoss += bceLoss(prediction, labels[idx])
            if ((prediction >= 0.5) == (labels[idx] >= 0.5)) correct++
        }
        return Evaluation(totalLoss / indices.size, correct.toDouble() / indices.size)
    }

    /**
     * Тестовые вероятности: по одному double на окно вместо 343 чисел в старом формате.
     * Массивы сортируются сразу — evaluateThresholds считает confusion matrix бинарным
     * поиском, а saveTo пишет их как есть.
     */
    private fun testProbabilities(
        state: Weights,
        features: DoubleArray,
        labels: DoubleArray,
        indices: IntArray,
        input: DoubleArray,
        buffers: Array<DoubleArray>
    ): Pair<DoubleArray, DoubleArray> {
        val legit = DoubleArray(indices.size)
        val cheat = DoubleArray(indices.size)
        var nLegit = 0
        var nCheat = 0
        for (idx in indices) {
            normalizeAt(features, idx, state, input)
            val p = forwardOnly(state, input, buffers)
            if (labels[idx] < 0.5) legit[nLegit++] = p else cheat[nCheat++] = p
        }
        java.util.Arrays.sort(legit, 0, nLegit)
        java.util.Arrays.sort(cheat, 0, nCheat)
        return legit.copyOf(nLegit) to cheat.copyOf(nCheat)
    }

    /** Метрики по уже посчитанным вероятностям (порядок не важен, но импорт детерминирован). */
    private fun metricsOf(legit: DoubleArray, cheat: DoubleArray): Evaluation {
        val total = legit.size + cheat.size
        if (total == 0) return Evaluation(0.0, 0.0)
        var totalLoss = 0.0
        var correct = 0
        for (p in legit) {
            totalLoss += bceLoss(p, 0.0)
            if (p < 0.5) correct++
        }
        for (p in cheat) {
            totalLoss += bceLoss(p, 1.0)
            if (p >= 0.5) correct++
        }
        return Evaluation(totalLoss / total, correct.toDouble() / total)
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
     * Вероятности уже посчитаны обучением и отсортированы, поэтому каждый порог
     * обрабатывается бинарным поиском — ни forward-pass, ни прохода по всем окнам.
     * @return null, если в этом сеансе ещё не было train() и model.dat без test-блока
     */
    fun evaluateThresholds(thresholds: DoubleArray): List<ThresholdMetrics>? {
        val testSet = lastTestSet ?: return null
        val legit = testSet.legitProbabilities
        val cheat = testSet.cheatProbabilities
        val totalLegit = legit.size
        val totalCheat = cheat.size

        /** Индекс первого элемента ≥ [value] в отсортированном массиве. */
        fun lowerBound(values: DoubleArray, value: Double): Int {
            var lo = 0
            var hi = values.size
            while (lo < hi) {
                val mid = (lo + hi) ushr 1
                if (values[mid] < value) lo = mid + 1 else hi = mid
            }
            return lo
        }

        return thresholds.map { threshold ->
            val tp = cheat.size - lowerBound(cheat, threshold)
            val fn = cheat.size - tp
            val fp = legit.size - lowerBound(legit, threshold)
            val tn = legit.size - fp
            ThresholdMetrics(threshold, tp, fp, tn, fn, totalLegit, totalCheat)
        }
    }

    private class IndexSplit(
        val train: IntArray,
        val validation: IntArray,
        val test: IntArray,
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
    private fun splitIndices(labels: DoubleArray, groups: Array<String?>?, rng: Random): IndexSplit {
        val train = ArrayList<Int>()
        val validation = ArrayList<Int>()
        val test = ArrayList<Int>()
        val grouped = BooleanArray(2)
        val groupCounts = IntArray(2)

        for ((c, label) in doubleArrayOf(0.0, 1.0).withIndex()) {
            val classIdx = (0 until labels.size).filter { labels[it] == label }
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
        return IndexSplit(train.toIntArray(), validation.toIntArray(), test.toIntArray(), summary)
    }

    /** He-инициализация весов (оптимальна для ReLU), bias и моменты Adam — нули. */
    private fun initWeights(state: Weights) {
        val rng = FastRandom(seed)
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

        if (ctx.fastMath) updateWeightsFast(state, ctx, features)
        else updateWeights(state, ctx, features)

        return prediction
    }

    /** Оригинальный шаг Adam: делим m и v на bias correction на каждом весе. */
    private fun updateWeights(state: Weights, ctx: StepContext, features: DoubleArray) {
        val learningRate = ctx.learningRate
        val l2Lambda = ctx.l2Lambda
        val bias1 = ctx.biasCorrection1
        val bias2 = ctx.biasCorrection2
        val layerCount = state.weights.size
        val activations = ctx.activations
        val deltas = ctx.deltas

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
    }

    /**
     * Опциональный (~system property daynac.training.fast-math) вариант шага Adam:
     * деление на bias correction вынесено из цикла по весам. Арифметика меняется
     * в последних битах, поэтому путь по умолчанию остаётся прежним.
     */
    private fun updateWeightsFast(state: Weights, ctx: StepContext, features: DoubleArray) {
        val lrOverBias1 = ctx.learningRate / ctx.biasCorrection1
        val invBias2 = 1.0 / ctx.biasCorrection2
        val l2Lambda = ctx.l2Lambda
        val layerCount = state.weights.size
        val activations = ctx.activations
        val deltas = ctx.deltas

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
                    val grad = Math.min(CLAMP, Math.max(-CLAMP, dj * prev[k] + l2Lambda * wj[k]))
                    val m = BETA1 * mj[k] + (1.0 - BETA1) * grad
                    val v = BETA2 * vj[k] + (1.0 - BETA2) * grad * grad
                    mj[k] = m
                    vj[k] = v
                    wj[k] -= lrOverBias1 * m / (sqrt(v * invBias2) + ADAM_EPS)
                }

                val mb = BETA1 * mBiases[j] + (1.0 - BETA1) * dj
                val vb = BETA2 * vBiases[j] + (1.0 - BETA2) * dj * dj
                mBiases[j] = mb
                vBiases[j] = vb
                biases[j] -= lrOverBias1 * mb / (sqrt(vb * invBias2) + ADAM_EPS)
            }
        }
    }

    private fun bceLoss(prediction: Double, label: Double): Double {
        val p = prediction.coerceIn(EPS, 1.0 - EPS)
        return -(label * ln(p) + (1.0 - label) * ln(1.0 - p))
    }

    private data class Evaluation(val avgLoss: Double, val accuracy: Double)

    /**
     * Z-score статистики по train-выборке; константный признак получает std = 1.0.
     * Читает плоский массив прямо по индексам train — лишних копий признаков нет.
     */
    private fun computeNormStats(state: Weights, features: DoubleArray, trainIndices: IntArray) {
        val n = trainIndices.size
        if (n == 0) return
        for (i in 0 until inputSize) {
            var mean = 0.0
            for (idx in trainIndices) mean += features[idx * inputSize + i]
            mean /= n

            var variance = 0.0
            for (idx in trainIndices) {
                val d = features[idx * inputSize + i] - mean
                variance += d * d
            }
            variance /= n

            val std = sqrt(variance)
            state.featureMean[i] = mean
            state.featureStd[i] = if (std < MIN_STD) 1.0 else std
        }
    }

    /**
     * Проверяет, что модель в файле совместима с текущим [layerSizes].
     */
    fun isCompatibleWith(file: File): Boolean {
        if (!file.exists()) return false
        return try {
            DataInputStream(file.inputStream().buffered()).use { input ->
                if (input.readInt() != MAGIC) return false
                if (input.readInt() != FORMAT_VERSION) return false
                if (input.readInt() != FeatureExtractor.FEATURE_VERSION) return false
                if (input.readInt() != inputSize) return false
                val sizes = layerSizes
                if (input.readInt() != sizes.size) return false
                // Полное сравнение размеров слоёв, а не только их количества:
                // [3,4,1] и [3,8,1] дают одинаковый размер массива, но несовместимы.
                for (size in sizes) {
                    if (input.readInt() != size) return false
                }
                true
            }
        } catch (_: Exception) {
            false
        }
    }

    /**
     * Слои в model.dat не совпадают с текущими [layerSizes]? Причины несовместимости
     * две: смена hidden-layers (файл снова станет совместимым, если вернуть старую
     * архитектуру — его нельзя переименовывать) и смена версии признаков/формата
     * (файл совместимым уже не будет никогда — его можно убирать с дороги).
     * Для файлов с другим входом/версией возвращает false.
     */
    fun isArchitectureMismatch(file: File): Boolean {
        if (!file.exists()) return false
        return try {
            DataInputStream(file.inputStream().buffered()).use { input ->
                input.readInt() // MAGIC
                input.readInt() // FORMAT_VERSION
                input.readInt() // FEATURE_VERSION
                input.readInt() // inputSize
                val count = input.readInt()
                val stored = IntArray(count) { input.readInt() }
                !stored.contentEquals(layerSizes)
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

                // --- Необязательный блок test-сета --------------------------------
                // Храним не сами окна, а только вероятности по каждому классу: 8 байт
                // на сэмпл вместо 8 * (inputSize + 1). Раньше блок рос вместе с датасетом
                // и для 100k окон добавлял ~275 МБ к model.dat.
                val ts = lastTestSet
                if (ts != null) {
                    output.writeBoolean(true)
                    output.writeInt(COMPACT_TEST_BLOCK)
                    output.writeInt(ts.legitProbabilities.size)
                    for (p in ts.legitProbabilities) output.writeDouble(p)
                    output.writeInt(ts.cheatProbabilities.size)
                    for (p in ts.cheatProbabilities) output.writeDouble(p)
                    // Разбиение датасета — для честного предупреждения в eval о
                    // случайном (не по группам) split после рестарта.
                    val sp = tr?.split
                    if (sp != null) {
                        output.writeBoolean(true)
                        output.writeBoolean(sp.legitGrouped)
                        output.writeBoolean(sp.cheatGrouped)
                        output.writeInt(sp.legitGroups)
                        output.writeInt(sp.cheatGroups)
                        output.writeInt(sp.trainSize)
                        output.writeInt(sp.valSize)
                        output.writeInt(sp.testSize)
                    } else {
                        output.writeBoolean(false)
                    }
                } else {
                    output.writeBoolean(false)
                }
            }
            try {
                Files.move(
                    tmpFile.toPath(), file.toPath(),
                    StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING
                )
            } catch (_: AtomicMoveNotSupportedException) {
                Files.move(tmpFile.toPath(), file.toPath(), StandardCopyOption.REPLACE_EXISTING)
            }
        } catch (e: Exception) {
            tmpFile.delete()
            throw e
        }
    }

    /**
     * Читает необязательный блок test-сета из model.dat (см. saveTo).
     * Возвращает Pair(testSet, splitSummary); для старых файлов без блока — (null, null).
     * Поддерживаются оба формата: новый компактный (маркер [COMPACT_TEST_BLOCK] и два
     * массива вероятностей) и старый (нормализованные окна + метки — probabilities
     * пересчитываются по загруженным весам). EOF внутри блока трактуется как отсутствие
     * блока: обрыв посреди записи невозможен, значит файл сохранён старой версией плагина.
     */
    private fun readTestSetBlock(input: DataInputStream, state: Weights): Pair<HoldoutSet?, SplitSummary?> {
        if (input.read() != 1) return null to null
        return try {
            val countOrMarker = input.readInt()
            val testSet = if (countOrMarker == COMPACT_TEST_BLOCK) {
                val nLegit = input.readInt()
                check(nLegit >= 0) { "Повреждённый блок test-сета (отрицательный размер legit: $nLegit)" }
                val legit = DoubleArray(nLegit) { input.readDouble() }
                val nCheat = input.readInt()
                check(nCheat >= 0) { "Повреждённый блок test-сета (отрицательный размер cheat: $nCheat)" }
                val cheat = DoubleArray(nCheat) { input.readDouble() }
                HoldoutSet(legit, cheat)
            } else {
                val count = countOrMarker
                check(count >= 0) { "Повреждённый блок test-сета (отрицательный размер: $count)" }
                val buffers = newLayerBuffers(state)
                val legitProbs = ArrayList<Double>(count)
                val cheatProbs = ArrayList<Double>(count)
                for (i in 0 until count) {
                    val sample = DoubleArray(inputSize)
                    for (k in sample.indices) sample[k] = input.readDouble()
                    val label = input.readDouble()
                    val probability = forwardOnly(state, sample, buffers)
                    if (label < 0.5) legitProbs.add(probability) else cheatProbs.add(probability)
                }
                HoldoutSet(legitProbs.sorted().toDoubleArray(), cheatProbs.sorted().toDoubleArray())
            }
            val split = if (input.read() == 1) {
                SplitSummary(
                    legitGrouped = input.readBoolean(),
                    cheatGrouped = input.readBoolean(),
                    legitGroups = input.readInt(),
                    cheatGroups = input.readInt(),
                    trainSize = input.readInt(),
                    valSize = input.readInt(),
                    testSize = input.readInt()
                )
            } else null
            testSet to split
        } catch (_: EOFException) {
            null to null
        }
    }

    fun loadFrom(file: File) {
        mutationLock.withLock {
            try {
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
                    val metricsFlag = input.read()
                    val trainResult: TrainResult? = when (metricsFlag) {
                        -1, 0 -> null
                        else -> try {
                            TrainResult(
                                epochsRun = input.readInt(),
                                bestEpoch = input.readInt(),
                                earlyStopped = input.readBoolean(),
                                trainLoss = input.readDouble(),
                                valLoss = input.readDouble(),
                                trainAccuracy = input.readDouble(),
                                valAccuracy = input.readDouble()
                            )
                        } catch (e: EOFException) {
                            throw IllegalStateException("Файл модели обрезан: блок метрик обучения неполный", e)
                        }
                    }

                    lastTrainResult = trainResult

                    // --- Необязательный блок test-сета ------------------------------
                    // Есть в файлах, сохранённых текущей версией; старые файлы
                    // (FORMAT_VERSION 3 без test-блока) заканчиваются после блока
                    // метрик — read() вернёт -1, и это не ошибка. Восстанавливаем
                    // lastTestSet, чтобы /daynac eval работал после рестарта.
                    val (testSet, testSplit) = readTestSetBlock(input, loaded)
                    lastTestSet = testSet

                    // testLoss/testAccuracy/split не писались в файл: пересчитываем
                    // по восстановленным весам и test-сету (в model.dat лежат те же
                    // лучшие веса, при которых train считал финальные метрики) —
                    // /daynac info показывает полную сводку и после рестарта.
                    lastTrainResult = trainResult?.let { tr ->
                        val eval = testSet?.let { metricsOf(it.legitProbabilities, it.cheatProbabilities) }
                        tr.copy(
                            testLoss = eval?.avgLoss,
                            testAccuracy = eval?.accuracy,
                            split = testSplit ?: tr.split
                        )
                    }
                    snapshot.set(loaded)
                }
            } catch (e: EOFException) {
                throw IllegalStateException("Файл модели обрезан или повреждён (неожиданный конец данных)", e)
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
