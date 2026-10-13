package zov.grusha.daynAC.exemptions

import org.bukkit.configuration.file.YamlConfiguration
import java.io.File
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap

/**
 * Временные исключения игроков из детекции: стримеры, тестеры, доверенные.
 *
 * Не путать с правом `daynac.exempt`: право выдаётся статически через LuckPerms
 * и бессрочно, а записи здесь — динамические, с опциональным сроком и снятием
 * через /daynac exempt <ник> off.
 *
 * Хранение: `exemptions.yml` в dataFolder. Ключ — UUID, значение — время истечения
 * в мс (Long.MAX_VALUE = бессрочно). Файл пишется при каждом изменении; просроченные
 * записи удаляются лениво при isExempt() и периодически через cleanupExpired().
 *
 * Потокобезопасность: карта — ConcurrentHashMap; save() сериализован отдельным
 * локом, чтобы параллельные exempt/unexempt не писали файл одновременно.
 */
class ExemptionManager(private val dataFolder: File) {

    private val file = File(dataFolder, "exemptions.yml")
    private val exemptions = ConcurrentHashMap<UUID, Long>()
    private val saveLock = Any()

    init {
        load()
    }

    /**
     * Освобождён ли игрок в данный момент. Побочно удаляет просроченную запись,
     * чтобы карта не росла от «мёртвых» исключений, если cleanupExpired давно
     * не вызывался.
     */
    fun isExempt(playerId: UUID): Boolean {
        val expiry = exemptions[playerId] ?: return false
        if (expiry == PERMANENT) return true
        if (System.currentTimeMillis() >= expiry) {
            exemptions.remove(playerId, expiry) // атомарно только если не перезаписали
            return false
        }
        return true
    }

    /**
     * Выдать или продлить исключение. [durationMillis] <= 0 = бессрочно.
     * Возвращает итоговое время истечения (PERMANENT для бессрочного).
     */
    fun exempt(playerId: UUID, durationMillis: Long): Long {
        val expiry = if (durationMillis <= 0) PERMANENT
        else System.currentTimeMillis() + durationMillis
        exemptions[playerId] = expiry
        save()
        return expiry
    }

    /** Снять исключение. true, если запись была и удалена. */
    fun unexempt(playerId: UUID): Boolean {
        val removed = exemptions.remove(playerId) != null
        if (removed) save()
        return removed
    }

    /** Время истечения (мс) или null, если игрок не в списке. PERMANENT = бессрочно. */
    fun getExpiry(playerId: UUID): Long? = exemptions[playerId]

    /** Текущий размер списка (для /daynac info). */
    fun size(): Int = exemptions.size

    /** Удалить все просроченные записи. Вызывается периодически из onEnable. */
    fun cleanupExpired() {
        val now = System.currentTimeMillis()
        val removed = exemptions.entries.removeIf { it.value != PERMANENT && it.value <= now }
        if (removed) save()
    }

    private fun load() {
        if (!file.exists()) return
        val yaml = YamlConfiguration.loadConfiguration(file)
        val section = yaml.getConfigurationSection("exemptions") ?: return
        for (key in section.getKeys(false)) {
            val uuid = try {
                UUID.fromString(key)
            } catch (_: IllegalArgumentException) {
                System.err.println("[INFO] [exemptions.yml] Некорректный UUID: $key — пропущен")
                continue
            }
            val value = section.getLong(key, -1L)
            exemptions[uuid] = if (value <= 0) PERMANENT else value
        }
    }

    private fun save() {
        synchronized(saveLock) {
            val yaml = YamlConfiguration()
            for ((uuid, expiry) in exemptions) {
                yaml.set("exemptions.$uuid", expiry)
            }
            file.parentFile?.mkdirs()
            yaml.save(file)
        }
    }

    companion object {
        /** Маркер бессрочного исключения. */
        const val PERMANENT = Long.MAX_VALUE
    }
}
