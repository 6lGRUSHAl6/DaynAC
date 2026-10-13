package zov.grusha.daynAC.permissions

/**
 * Все права плагина в одном месте. Должны совпадать с plugin.yml.
 *
 * daynac.admin — родитель всех прав, КРОМЕ [EXEMPT]: иммунитет к анализу
 * выдаётся только явно, чтобы украденный админ-аккаунт не был неуязвим для античита.
 */
object Permissions {
    const val ADMIN = "daynac.admin"
    const val VIEW = "daynac.view"           // ver, info, allinfo (только просмотр)
    const val ALERTS = "daynac.alerts"       // оповещения и лог ударов в чат
    const val SPECTATE = "daynac.spectate"   // телепорт из GUI в режиме наблюдателя
    const val RECORD = "daynac.record"       // запись датасета (даёт игроку иммунитет на время сессии)
    const val TRAIN = "daynac.train"         // train, eval
    const val CONFIG = "daynac.config"       // reload, debug
    const val EXEMPT = "daynac.exempt"       // иммунитет к детекции

    /** Подкоманда -> право, необходимое для её выполнения. */
    val SUBCOMMANDS: Map<String, String> = linkedMapOf(
        "ver" to VIEW,
        "info" to VIEW,
        "allinfo" to VIEW,
        "record" to RECORD,
        "train" to TRAIN,
        "eval" to TRAIN,
        "reload" to CONFIG,
        "debug" to CONFIG
    )
}