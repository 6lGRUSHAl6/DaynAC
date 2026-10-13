package zov.grusha.daynAC.permissions

/**
 * Все права плагина в одном месте. Должны совпадать с plugin.yml.
 *
 * daynac.admin — родитель всех прав, КРОМЕ [EXEMPT]: OP по умолчанию
 * проверяется детекцией, чтобы на нём можно было тестировать фильтр без
 * второго аккаунта. Временные исключения ([EXEMPT_MANAGE]) в admin входят —
 * это управление, а не иммунитет.
 */
object Permissions {
    const val ADMIN = "daynac.admin"
    const val VIEW = "daynac.view"                    // ver, info, allinfo (только просмотр)
    const val ALERTS = "daynac.alerts"                // оповещения и лог ударов в чат
    const val SPECTATE = "daynac.spectate"            // телепорт из GUI в режиме наблюдателя
    const val RECORD = "daynac.record"                // запись датасета (даёт игроку иммунитет на время сессии)
    const val TRAIN = "daynac.train"                  // train, eval
    const val CONFIG = "daynac.config"                // reload, debug, clearflags
    const val EXEMPT = "daynac.exempt"                // иммунитет к детекции
    const val EXEMPT_MANAGE = "daynac.exempt.manage"  // выдача/снятие временных исключений

    /** Подкоманда -> право, необходимое для её выполнения. */
    val SUBCOMMANDS: Map<String, String> = linkedMapOf(
        "ver" to VIEW,
        "info" to VIEW,
        "allinfo" to VIEW,
        "record" to RECORD,
        "train" to TRAIN,
        "eval" to TRAIN,
        "clearflags" to CONFIG,
        "exempt" to EXEMPT_MANAGE,
        "reload" to CONFIG,
        "debug" to CONFIG
    )
}