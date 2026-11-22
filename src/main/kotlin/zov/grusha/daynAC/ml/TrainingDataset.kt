package zov.grusha.daynAC.ml

/**
 * Компактное представление векторного датасета для обучения.
 *
 * Раньше в память загружался `List<VectorSample>` (объект + `DoubleArray` на каждый
 * сэмпл), после чего `DaynAC` делал ещё два списка (`features`, `labels`) и третий
 * список групп, а `NeuralNetwork.train()` — ещё по копии каждого вектора на
 * train/val/test. На 100k окон это давало 4–5 копий признаков и 600–700 МБ пика.
 *
 * Здесь все признаки лежат в ОДНОМ плоском [DoubleArray]: сэмпл `i` занимает
 * диапазон `[i * featureCount, (i + 1) * featureCount)`. Метки и метаданные тоже
 * хранятся в массивах. Обучение нормализует признаки «на лету» в один буфер на
 * поток, так что вторая большая копия датасета не создаётся вовсе.
 *
 * [sessionIds] и [playerHashes] — метаданные для честного разбиения train/val/test;
 * в признаки они не входят. [groupsBy] выбирает нужную группировку по ключу
 * `training.split-by` (значение конфига читается на главном потоке до старта
 * фонового обучения и передаётся в него уже готовым значением).
 */
class TrainingDataset(
    val features: DoubleArray,
    val labels: DoubleArray,
    val sessionIds: Array<String?>,
    val playerHashes: Array<String?>,
    val sampleCount: Int,
    val legitCount: Int,
    val cheatCount: Int
) {
    init {
        require(sampleCount >= 0) { "sampleCount не может быть отрицательным" }
        require(labels.size == sampleCount) { "labels и sampleCount разной длины" }
        require(sessionIds.size == sampleCount) { "sessionIds и sampleCount разной длины" }
        require(playerHashes.size == sampleCount) { "playerHashes и sampleCount разной длины" }
        require(features.size % sampleCount == 0 || sampleCount == 0) {
            "features не делится на sampleCount без остатка"
        }
    }

    /** Число признаков в одном сэмпле (всегда делит [features] на [sampleCount]). */
    val featureCount: Int get() = if (sampleCount == 0) 0 else features.size / sampleCount

    /** Группы для разбиения: сессии записи или игроки. */
    fun groupsBy(splitBy: String): Array<String?> =
        if (splitBy == "player") playerHashes else sessionIds
}
