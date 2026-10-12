package zov.grusha.daynAC.compat

import net.kyori.adventure.text.serializer.legacy.LegacyComponentSerializer
import org.bukkit.entity.Player

/**
 * Один jar работает на Paper 1.21.11 и новее (26.1, 26.2, ...), хотя собран против API 1.21.11.
 * Для большинства вызовов Paper сохраняет бинарную совместимость, но между версиями меняются
 * внешние библиотеки (например, Adventure 4.x -> 5.x). Если вызов вдруг не находится в рантайме
 * (NoSuchMethodError / NoClassDefFoundError — оба наследники [LinkageError]), эти обёртки
 * переключаются на запасной путь, а не роняют обработчик события. Только для косметики
 * и вспомогательных вызовов — основная логика детекции от них не зависит.
 */
object Compat {

    @Volatile private var adventureActionBar = true
    @Volatile private var legacyActionBar = true

    /**
     * Сообщение над хотбаром с legacy-цветами (§). Вызывать с главного потока.
     * Сначала Adventure, при несовместимости — строковый Bukkit API, иначе тихо ничего.
     */
    fun actionBar(player: Player, text: String) {
        if (adventureActionBar) {
            try {
                player.sendActionBar(LegacyComponentSerializer.legacySection().deserialize(text))
                return
            } catch (e: LinkageError) {
                adventureActionBar = false // дальше не пытаемся
            }
        }
        if (legacyActionBar) {
            try {
                @Suppress("DEPRECATION")
                player.sendActionBar(text)
            } catch (e: LinkageError) {
                legacyActionBar = false
            }
        }
    }
}