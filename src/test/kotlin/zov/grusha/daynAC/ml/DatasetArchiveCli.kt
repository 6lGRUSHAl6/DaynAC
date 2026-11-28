package zov.grusha.daynAC.ml

import java.io.File
import kotlin.system.exitProcess

/**
 * Headless-утилита для переноса датасета без запуска сервера. Используется теми же
 * Gradle-задачами, что и benchTrain: тот же код архиватора, те же проверки.
 *
 *   ./gradlew datasetExport --args="run/plugins/DaynAC run/dataset.dac"
 *   ./gradlew datasetVerify  --args="run/dataset.dac"
 *   ./gradlew datasetImport  --args="run/dataset.dac run/plugins/DaynAC --replace-salt"
 *
 * Утилита не координируется с запущенным сервером (мьютекс датасета живёт внутри его JVM):
 * export/import выполняйте при остановленном сервере.
 */
private const val WINDOW_SIZE = 16 // источник правды — DaynAC.WINDOW_SIZE; не тянем Bukkit в CLI

/** Позиционные аргументы и флаги (`--…`) раздельно: флаг в любом месте не должен становиться путём. */
internal class CliArgs(val positional: List<String>, val flags: Set<String>)

internal fun parseCliArgs(rest: List<String>): CliArgs =
    CliArgs(
        positional = rest.filterNot { it.startsWith("--") },
        flags = rest.filter { it.startsWith("--") }.toSet()
    )

fun main(args: Array<String>) {
    // Gradle-задачи уже выбирают действие через system property; явный первый аргумент
    // тоже поддерживается, чтобы утилиту можно было вызвать напрямую.
    val explicit = args.getOrNull(0)?.lowercase()?.takeIf { it in setOf("export", "verify", "import") }
    val action = explicit ?: System.getProperty("daynac.dataset.action")
    if (action == null) {
        usage()
        exitProcess(2)
    }
    val parsed = parseCliArgs((if (explicit != null) args.copyOfRange(1, args.size) else args).toList())
    val pos = parsed.positional
    val flags = parsed.flags

    try {
        when (action) {
            "export" -> {
                if (!flags.all { it == "--no-salt" } || pos.size > 2) { usage(); exitProcess(2) }
                val dataDir = File(pos.getOrNull(0) ?: "run/plugins/DaynAC")
                val archive = pos.getOrNull(1)?.let(::File) ?: File(dataDir, "dataset.dac")
                val result = DatasetArchive(DatasetManager(dataDir, WINDOW_SIZE)).export(
                    archive, DatasetArchive.ExportOptions(includeSalt = "--no-salt" !in flags)
                )
                println(
                    "Архив ${result.file}: ${result.bytes} байт, ${result.numericRows} числовых / " +
                            "${result.rawLines} служебных строк, соль ${if (result.includedSalt) "включена" else "исключена"}"
                )
                if (result.includedSalt) println("ВНИМАНИЕ: архив содержит dataset.salt — не публикуйте его")
            }
            "verify" -> {
                if (flags.isNotEmpty() || pos.size > 1) { usage(); exitProcess(2) }
                val archive = File(pos.getOrNull(0) ?: "run/dataset.dac")
                // DatasetArchive привязан к DatasetManager; для проверки создаём временный каталог.
                val work = File(archive.absoluteFile.parentFile, ".dac-cli-work")
                try {
                    val info = DatasetArchive(DatasetManager(work, WINDOW_SIZE)).verify(archive)
                    println(
                        "признаки v${info.featureVersion}, вход ${info.featureCount}, " +
                                "строк ${info.legitRows}/${info.cheatRows} (legit/cheat), " +
                                "${info.numericRows} числовых / ${info.rawLines} служебных, " +
                                "соль ${if (info.includedSalt) "есть" else "нет"}"
                    )
                } finally {
                    work.deleteRecursively()
                }
            }
            "import" -> {
                if (!flags.all { it == "--replace-salt" } || pos.size > 2) { usage(); exitProcess(2) }
                val archive = File(pos.getOrNull(0) ?: "run/dataset.dac")
                val dataDir = File(pos.getOrNull(1) ?: "run/plugins/DaynAC")
                val result = DatasetArchive(DatasetManager(dataDir, WINDOW_SIZE)).import(
                    archive, DatasetArchive.ImportOptions(replaceSalt = "--replace-salt" in flags)
                )
                println(
                    "Восстановлено ${result.numericRows} числовых / ${result.rawLines} служебных строк. " +
                            "Резервные копии: ${result.backups.joinToString { it.name }}"
                )
            }
            else -> { usage(); exitProcess(2) }
        }
    } catch (e: Exception) {
        // Без stack trace: причина (битый архив, несовместимая версия, чужая соль) понятна из текста.
        System.err.println("Ошибка: ${e.message ?: e.javaClass.simpleName}")
        exitProcess(1)
    }
}

private fun usage() {
    println(
        """
        Использование:
          export <каталог_датасета> [путь_архива] [--no-salt]
          verify <путь_архива>
          import <путь_архива> <каталог_датасета> [--replace-salt]
        """.trimIndent()
    )
}