package zov.grusha.daynAC.exemptions

import org.bukkit.configuration.file.YamlConfiguration
import java.io.File
import java.io.IOException
import java.nio.charset.StandardCharsets
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.StandardCopyOption
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
        val yaml = YamlConfiguration()
        try {
            // loadConfiguration(File) глотает InvalidConfigurationException и отдаёт пустой
            // конфиг — битый файл не отличить от пустого, а следующий save() затёр бы его.
            // Поэтому читаем через load(File), где ошибка видна.
            yaml.load(file)
        } catch (e: org.bukkit.configuration.InvalidConfigurationException) {
            quarantineCorruptFile(e)
            return
        } catch (e: java.io.IOException) {
            quarantineCorruptFile(e)
            return
        }
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

    /**
     * Битый файл не должен ронять onEnable и не должен быть молча перезаписан:
     * переименовываем его в `exemptions.yml.corrupt-<время>` для ручного разбора
     * и стартуем с пустым списком.
     */
    private fun quarantineCorruptFile(cause: Exception) {
        val backup = File(file.parentFile, "${file.name}.corrupt-${System.currentTimeMillis()}")
        val moved = try {
            Files.move(file.toPath(), backup.toPath(), StandardCopyOption.REPLACE_EXISTING)
            true
        } catch (_: IOException) {
            false
        }
        System.err.println(
            "[WARN] [exemptions.yml] Файл повреждён (${cause.message}); " +
                    (if (moved) "сохранён как ${backup.name}" else "переименовать не удалось") +
                    ", список исключений пуст"
        )
    }

    /**
     * Атомарная запись: сериализуем в строку, пишем во временный файл рядом и
     * переименовываем поверх основного. Обрыв посреди записи оставляет старый
     * exemptions.yml нетронутым (в худшем случае — лишний .tmp).
     */
    private fun save() {
        synchronized(saveLock) {
            val yaml = YamlConfiguration()
            for ((uuid, expiry) in exemptions) {
                yaml.set("exemptions.$uuid", expiry)
            }
            val data = yaml.saveToString()
            file.parentFile?.mkdirs()
            val tmp = File(file.parentFile, "${file.name}.tmp")
            try {
                Files.write(tmp.toPath(), data.toByteArray(StandardCharsets.UTF_8))
                try {
                    Files.move(
                        tmp.toPath(), file.toPath(),
                        StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING
                    )
                } catch (_: AtomicMoveNotSupportedException) {
                    Files.move(tmp.toPath(), file.toPath(), StandardCopyOption.REPLACE_EXISTING)
                }
            } catch (e: IOException) {
                tmp.delete()
                System.err.println("[WARN] [exemptions.yml] Не удалось сохранить: ${e.message}")
            }
        }
    }

    companion object {
        /** Маркер бессрочного исключения. */
        const val PERMANENT = Long.MAX_VALUE
    }
}
