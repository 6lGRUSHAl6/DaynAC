package zov.grusha.daynAC.ml

import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.util.Random
import java.util.concurrent.locks.ReentrantReadWriteLock
import kotlin.concurrent.read
import kotlin.concurrent.write
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * MLP-классификатор «читер / легит» на чистом Kotlin, без внешних ML-библиотек.
 *
 * Архитектура: 327 → 128 → 64 → 32 → 1
 * (327 = 16 ударов × 20 признаков + 7 агрегатов из [FeatureExtractor]).
 * Скрытые слои — ReLU, выход — Sigmoid (вероятность чита от 0 до 1).
 *
 * Обучение: Binary Cross-Entropy + backpropagation + SGD (по одному сэмплу).
 *
 * Защита от переобучения (конструктивная, а не только подбором гиперпараметров):
 *  1. Узкий «бутылочный» хвост 64 → 32 → 1 — сжатие размерности перед выходом
 *     ограничивает ёмкость сети запоминать шум.
 *  2. L2 weight decay — каждое обновление весов штрафуется пропорционально
 *     величине веса, сеть стремится к малым весам => более гладкая граница решений.
 *  3. Early stopping на валидационной выборке (20%) с восстановлением лучших
 *     весов — обучение останавливается до того, как сеть начнёт заучивать train.
 *  4. Z-score нормализация входов — без неё большие по масштабу признаки
 *     (distance, hitTimeDelta) приводят к гигантским градиентам и нестабильному
 *     обучению, которое часто «лечат» завышенным lr и переобучают.
 *
 * Числовые защиты (как в daynac.md): clipping градиентов и дельт ±5.0,
 * clamp входа sigmoid в [-500, 500], epsilon 1e-15 в логарифмах BCE.
 *
 * Потокобезопасность: predict/save под read-lock, train под write-lock.
 */
class NeuralNetwork(
    val inputSize: Int = 327, // = DaynAC.WINDOW_SIZE * FeatureExtractor.PER_HIT_FEATURES + FeatureExtractor.AGGREGATE_FEATURES
    hiddenSizes: IntArray = intArrayOf(128, 64, 32),
    private val seed: Long = 42L
) {

    /** Размеры всех слоёв: [inputSize, 128, 64, 32, 1]. */
    val layerSizes: IntArray = intArrayOf(inputSize, *hiddenSizes, 1)

    // weights[l][j][k]: связь k-го нейрона слоя l-1 с j-м нейроном слоя l
    private val weights: Array<Array<DoubleArray>> =
        Array(layerSizes.size - 1) { l ->
            Array(layerSizes[l + 1]) { DoubleArray(layerSizes[l]) }
        }

    private val biases: Array<DoubleArray> =
        Array(layerSizes.size - 1) { l -> DoubleArray(layerSizes[l + 1]) }

    /** Статистики Z-score, вычисляются при обучении и сериализуются вместе с моделью. */
    private var featureMean: DoubleArray = DoubleArray(inputSize)
    private var featureStd: DoubleArray = DoubleArray(inputSize) { 1.0 }

    private val lock = ReentrantReadWriteLock()

    companion object {
        private const val EPS = 1e-15          // защита log(0) в BCE
        private const val CLAMP = 5.0          // clipping градиентов и дельт
        private const val SIGMOID_CLAMP = 500.0
        private const val MIN_STD = 1e-8       // защита от деления на 0 в z-score
        private const val MAGIC = 0x44414E41   // "DANA" — маркер файла модели
        private const val FORMAT_VERSION = 1

        /** Разделение train/validation при обучении. */
        private const val VAL_FRACTION = 0.2
    }

    // ------------------------------------------------------------------
    // Активации
    // ------------------------------------------------------------------

    private fun relu(x: Double): Double = if (x > 0.0) x else 0.0

    private fun sigmoid(x: Double): Double {
        val clamped = x.coerceIn(-SIGMOID_CLAMP, SIGMOID_CLAMP)
        return 1.0 / (1.0 + exp(-clamped))
    }

    // ------------------------------------------------------------------
    // Forward pass
    // ------------------------------------------------------------------

    /**
     * Предсказание по «сырому» вектору признаков (без нормализации снаружи —
     * Z-score применяется внутри по сохранённым статистикам).
     * @return вероятность чита в (0, 1)
     */
    fun predict(rawFeatures: DoubleArray): Double {
        require(rawFeatures.size == inputSize) {
            "Ожидался вектор из $inputSize признаков, получен ${rawFeatures.size}"
        }

        lock.read {
            val normalized = normalize(rawFeatures)
            var activation = normalized

            for (l in weights.indices) {
                val next = DoubleArray(layerSizes[l + 1])
                val isLast = (l == weights.size - 1)
                for (j in next.indices) {
                    var z = biases[l][j]
                    val wj = weights[l][j]
                    for (k in activation.indices) {
                        z += wj[k] * activation[k]
                    }
                    next[j] = if (isLast) sigmoid(z) else relu(z)
                }
                activation = next
            }
            return activation[0]
        }
    }

    private fun normalize(features: DoubleArray): DoubleArray {
        val out = DoubleArray(features.size)
        for (i in features.indices) {
            out[i] = (features[i] - featureMean[i]) / featureStd[i]
        }
        return out
    }

    // ------------------------------------------------------------------
    // Обучение
    // ------------------------------------------------------------------

    /** Результат обучения — для отчёта администратору и диагностики переобучения. */
    data class TrainResult(
        val epochsRun: Int,
        val bestEpoch: Int,
        val earlyStopped: Boolean,
        val trainLoss: Double,
        val valLoss: Double,
        val trainAccuracy: Double,
        val valAccuracy: Double
    ) {
        /** Разрыв train/val — грубая метрика переобучения (0.05+ уже подозрительно). */
        val overfitGap: Double get() = trainAccuracy - valAccuracy
    }

    /**
     * Полный цикл обучения: split 80/20, нормализация по train-части,
     * SGD по одному сэмплу, early stopping с восстановлением лучших весов.
     *
     * @param samples векторы признаков (каждый длиной [inputSize])
     * @param labels  0.0 = легит, 1.0 = чит
     * @param learningRate шаг SGD (оптимально 0.001–0.003)
     * @param maxEpochs максимальное число эпох
     * @param patience сколько проверок val loss без улучшения до остановки
     * @param l2Lambda коэффициент L2 weight decay (0 = отключить; 1e-4 — разумный дефолт)
     * @param onProgress опциональный колбэк прогресса (эпоха, trainLoss, valLoss, valAcc)
     */
    fun train(
        samples: List<DoubleArray>,
        labels: List<Double>,
        learningRate: Double = 0.001,
        maxEpochs: Int = 1000,
        patience: Int = 15,
        l2Lambda: Double = 1e-4,
        onProgress: ((epoch: Int, trainLoss: Double, valLoss: Double?, valAcc: Double?) -> Unit)? = null
    ): TrainResult {
        require(samples.size == labels.size) { "samples и labels разной длины" }
        require(samples.size >= 10) { "Датасет слишком мал: ${samples.size} сэмплов (нужно ≥ 10)" }
        require(samples.all { it.size == inputSize }) { "Все сэмплы должны быть длиной $inputSize" }

        lock.write {
            // 1. Перемешиваем с фиксированным seed и делим 80/20
            val indices = samples.indices.shuffled(Random(seed))
            val valCount = max(1, (samples.size * VAL_FRACTION).toInt().coerceAtMost(indices.size / 2))
            val valIdx = indices.take(valCount)
            val trainIdx = indices.drop(valCount)

            // 2. Инициализация весов заново — чистый старт (He init)
            initWeights()

            // 3. Нормализационные статистики — ТОЛЬКО по train-части (без утечки данных)
            computeNormStats(trainIdx.map { samples[it] })
            val trainFeatures = trainIdx.map { normalize(samples[it]) }
            val valFeatures = valIdx.map { normalize(samples[it]) }
            val trainLabels = trainIdx.map { labels[it] }
            val valLabels = valIdx.map { labels[it] }

            // 4. Цикл обучения
            val valCheckInterval = max(1, min(5, maxEpochs / 50))
            var bestValLoss = Double.POSITIVE_INFINITY
            var bestEpoch = 0
            var staleCount = 0
            var earlyStopped = false
            var bestWeights = copyWeights()
            var bestBiases = copyBiases()

            var epoch = 0
            val epochOrder = ArrayList<Int>(trainFeatures.size)
            while (epoch < maxEpochs) {
                // SGD: перемешиваем порядок сэмплов каждую эпоху
                epochOrder.clear()
                for (i in trainFeatures.indices) epochOrder.add(i)
                epochOrder.shuffle(Random(seed + epoch + 1))

                var epochLoss = 0.0
                for (i in epochOrder) {
                    val prediction = forwardAndBackprop(trainFeatures[i], trainLabels[i], learningRate, l2Lambda)
                    epochLoss += bceLoss(prediction, trainLabels[i])
                }
                val trainLoss = epochLoss / trainFeatures.size

                // Проверка валидации каждые valCheckInterval эпох
                if (epoch % valCheckInterval == 0) {
                    val valResult = evaluateNormalized(valFeatures, valLabels)
                    onProgress?.invoke(epoch, trainLoss, valResult.avgLoss, valResult.accuracy)

                    if (valResult.avgLoss < bestValLoss) {
                        bestValLoss = valResult.avgLoss
                        bestEpoch = epoch
                        staleCount = 0
                        bestWeights = copyWeights()
                        bestBiases = copyBiases()
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

            // 5. Восстанавливаем лучшие веса — не последнюю (возможно переобученную) версию
            restoreWeights(bestWeights, bestBiases)

            // 6. Финальные метрики
            val finalTrain = evaluateNormalized(trainFeatures, trainLabels)
            val finalVal = evaluateNormalized(valFeatures, valLabels)

            return TrainResult(
                epochsRun = epoch,
                bestEpoch = bestEpoch,
                earlyStopped = earlyStopped,
                trainLoss = finalTrain.avgLoss,
                valLoss = finalVal.avgLoss,
                trainAccuracy = finalTrain.accuracy,
                valAccuracy = finalVal.accuracy
            )
        }
    }

    /** He-инициализация (2015) — оптимальна для ReLU; bias нулями. */
    private fun initWeights() {
        val rng = Random(seed)
        for (l in weights.indices) {
            val scale = sqrt(2.0 / layerSizes[l]) // fan-in слоя l
            for (j in weights[l].indices) {
                biases[l][j] = 0.0
                for (k in weights[l][j].indices) {
                    weights[l][j][k] = rng.nextGaussian() * scale
                }
            }
        }
    }

    /**
     * Forward + backward + SGD-обновление по одному сэмплу.
     * Дельта выходного слоя для BCE+Sigmoid упрощается до (ŷ − y).
     * Градиент веса: δ_j · a_k; обновление: w -= lr · (clip(grad) + l2 · w).
     * @return предсказание до обновления (для подсчёта loss)
     */
    private fun forwardAndBackprop(
        features: DoubleArray,
        label: Double,
        learningRate: Double,
        l2Lambda: Double
    ): Double {
        val layerCount = weights.size

        // Forward: сохраняем pre-activation (z) и activation (a) для backward
        val activations = arrayOfNulls<DoubleArray>(layerCount + 1)
        activations[0] = features
        for (l in 0 until layerCount) {
            val isLast = (l == layerCount - 1)
            val z = DoubleArray(layerSizes[l + 1])
            val a = DoubleArray(layerSizes[l + 1])
            val prev = activations[l]!!
            for (j in z.indices) {
                var sum = biases[l][j]
                val wj = weights[l][j]
                for (k in prev.indices) {
                    sum += wj[k] * prev[k]
                }
                z[j] = sum
                a[j] = if (isLast) sigmoid(sum) else relu(sum)
            }
            activations[l + 1] = a
            // z храним в отдельном локальном массиве только для текущего слоя —
            // для backward нужна только информация «активен ли ReLU», она выводится из a[j] > 0
        }

        val output = activations[layerCount]!![0]

        // Backward: дельты по слоям (от последнего к первому)
        val deltas = arrayOfNulls<DoubleArray>(layerCount)
        val lastDeltas = DoubleArray(1)
        lastDeltas[0] = (output - label).coerceIn(-CLAMP, CLAMP)
        deltas[layerCount - 1] = lastDeltas

        for (l in layerCount - 2 downTo 0) {
            val delta = DoubleArray(layerSizes[l + 1])
            val nextDelta = deltas[l + 1]!!
            val nextWeights = weights[l + 1]
            val activation = activations[l + 1]!!
            for (j in delta.indices) {
                var sum = 0.0
                for (n in nextDelta.indices) {
                    sum += nextWeights[n][j] * nextDelta[n]
                }
                // ReLU': 1 если нейрон был активен (a > 0), иначе 0
                delta[j] = if (activation[j] > 0.0) sum.coerceIn(-CLAMP, CLAMP) else 0.0
            }
            deltas[l] = delta
        }

        // SGD-обновление: w -= lr * (clip(δ·a + l2·w))
        for (l in 0 until layerCount) {
            val prev = activations[l]!!
            val delta = deltas[l]!!
            for (j in weights[l].indices) {
                val dj = delta[j]
                val wj = weights[l][j]
                for (k in wj.indices) {
                    val grad = (dj * prev[k] + l2Lambda * wj[k]).coerceIn(-CLAMP, CLAMP)
                    wj[k] -= learningRate * grad
                }
                biases[l][j] -= learningRate * dj.coerceIn(-CLAMP, CLAMP)
            }
        }

        return output
    }

    /** BCE для одного сэмпла. */
    private fun bceLoss(prediction: Double, label: Double): Double {
        val p = prediction.coerceIn(EPS, 1.0 - EPS)
        return -(label * ln(p) + (1.0 - label) * ln(1.0 - p))
    }

    /** Оценка на уже нормализованных сэмплах: средний BCE и accuracy (порог 0.5). */
    private fun evaluateNormalized(features: List<DoubleArray>, labels: List<Double>): Evaluation {
        if (features.isEmpty()) return Evaluation(0.0, 0.0)
        var totalLoss = 0.0
        var correct = 0
        for (i in features.indices) {
            val prediction = forwardOnly(features[i])
            totalLoss += bceLoss(prediction, labels[i])
            val predictedClass = if (prediction >= 0.5) 1.0 else 0.0
            if (predictedClass == labels[i]) correct++
        }
        return Evaluation(totalLoss / features.size, correct.toDouble() / features.size)
    }

    private data class Evaluation(val avgLoss: Double, val accuracy: Double)

    /** Прямой проход по уже нормализованным признакам (без взятия lock — вызывается под write-lock). */
    private fun forwardOnly(normalizedFeatures: DoubleArray): Double {
        var activation = normalizedFeatures
        for (l in weights.indices) {
            val next = DoubleArray(layerSizes[l + 1])
            val isLast = (l == weights.size - 1)
            for (j in next.indices) {
                var z = biases[l][j]
                val wj = weights[l][j]
                for (k in activation.indices) {
                    z += wj[k] * activation[k]
                }
                next[j] = if (isLast) sigmoid(z) else relu(z)
            }
            activation = next
        }
        return activation[0]
    }

    /** Z-score статистики по train-выборке. Константный признак получает std = 1.0. */
    private fun computeNormStats(trainFeatures: List<DoubleArray>) {
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

            featureMean[i] = mean
            featureStd[i] = if (sqrt(variance) < MIN_STD) 1.0 else sqrt(variance)
        }
    }

    // ------------------------------------------------------------------
    // Снапшоты весов (для early stopping)
    // ------------------------------------------------------------------

    private fun copyWeights(): Array<Array<DoubleArray>> =
        weights.map { layer -> layer.map { it.copyOf() }.toTypedArray() }.toTypedArray()

    private fun copyBiases(): Array<DoubleArray> = biases.map { it.copyOf() }.toTypedArray()

    private fun restoreWeights(w: Array<Array<DoubleArray>>, b: Array<DoubleArray>) {
        for (l in weights.indices) {
            for (j in weights[l].indices) {
                System.arraycopy(w[l][j], 0, weights[l][j], 0, w[l][j].size)
            }
            System.arraycopy(b[l], 0, biases[l], 0, b[l].size)
        }
    }

    // ------------------------------------------------------------------
    // Сохранение / загрузка (бинарный формат, без Java Serialization)
    // ------------------------------------------------------------------

    /** Проверяет, что модель в файле совместима с текущим [inputSize]. */
    fun isCompatibleWith(file: File): Boolean {
        if (!file.exists()) return false
        return try {
            DataInputStream(file.inputStream().buffered()).use { input ->
                input.readInt() == MAGIC &&
                    input.readInt() == FORMAT_VERSION &&
                    input.readInt() == inputSize &&
                    input.readInt() == layerSizes.size
            }
        } catch (_: Exception) {
            false
        }
    }

    fun saveTo(file: File) {
        lock.read {
            DataOutputStream(file.outputStream().buffered()).use { output ->
                output.writeInt(MAGIC)
                output.writeInt(FORMAT_VERSION)
                output.writeInt(inputSize)
                output.writeInt(layerSizes.size)
                for (size in layerSizes) output.writeInt(size)

                for (layer in weights) {
                    for (neuron in layer) {
                        for (w in neuron) output.writeDouble(w)
                    }
                }
                for (layer in biases) {
                    for (b in layer) output.writeDouble(b)
                }
                for (m in featureMean) output.writeDouble(m)
                for (s in featureStd) output.writeDouble(s)
            }
        }
    }

    fun loadFrom(file: File) {
        lock.write {
            DataInputStream(file.inputStream().buffered()).use { input ->
                check(input.readInt() == MAGIC) { "Файл модели повреждён (неверный маркер)" }
                check(input.readInt() == FORMAT_VERSION) { "Неподдерживаемая версия формата модели" }
                check(input.readInt() == inputSize) { "Размер входа модели не совпадает с текущим ($inputSize)" }
                val layerCount = input.readInt()
                check(layerCount == layerSizes.size) { "Структура слоёв модели не совпадает" }
                for (l in layerSizes.indices) {
                    check(input.readInt() == layerSizes[l]) { "Размер слоя $l не совпадает" }
                }

                for (layer in weights) {
                    for (neuron in layer) {
                        for (k in neuron.indices) neuron[k] = input.readDouble()
                    }
                }
                for (layer in biases) {
                    for (j in layer.indices) layer[j] = input.readDouble()
                }
                for (i in featureMean.indices) featureMean[i] = input.readDouble()
                for (i in featureStd.indices) featureStd[i] = input.readDouble()
            }
        }
    }
}
