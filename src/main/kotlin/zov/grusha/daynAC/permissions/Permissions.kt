package zov.grusha.daynAC.permissions

/**
 * Все права плагина в одном месте. Должны совпадать с plugin.yml.
 *
 * daynac.admin — родитель всех прав, КРОМЕ [EXEMPT]: иммунитет к анализу
 * выдаётся только явно, чтобы украденный админ-аккаунт не был неуязвим для античита.
 */
object Permissions {
    const val ADMIN = "daynac.admin"
    const val VIEW = "daynac.view"
    const val ALERTS = "daynac.alerts"
    const val SPECTATE = "daynac.spectate"
    const val RECORD = "daynac.record"
    const val TRAIN = "daynac.train"
    const val CONFIG = "daynac.config"
    const val EXEMPT = "daynac.exempt"                 // иммунитет (себе, через LuckPerms)
    const val EXEMPT_MANAGE = "daynac.exempt.manage"   // выдача/снятие временных исключений

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