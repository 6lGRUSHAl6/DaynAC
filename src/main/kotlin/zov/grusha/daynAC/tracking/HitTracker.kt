package zov.grusha.daynAC.tracking

import org.bukkit.entity.Player
import java.util.ArrayDeque
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap


/**
 * Хранит последние [maxSize] ударов игрока (по умолчанию = WINDOW_SIZE,
 * см. [zov.grusha.daynAC.DaynAC]).
 *
 * ВАЖНО: методы должны вызываться ТОЛЬКО из главного потока Bukkit.
 * ConcurrentHashMap защищает структуру мапы, но ArrayDeque внутри —
 * не потокобезопасен: одновременные addHit из разных потоков дадут
 * гонку на removeFirst/addLast. Если понадобится async-доступ — заменить
 * на ConcurrentLinkedDeque или добавить явную блокировку.
 */
class HitTracker(private val maxSize: Int) {
    private val history = ConcurrentHashMap<UUID, ArrayDeque<HitData>>()

    fun addHit(player: Player, hit: HitData) {
        val deque = history.computeIfAbsent(player.uniqueId) { ArrayDeque() }
        if (deque.size >= maxSize) deque.removeFirst()
        deque.addLast(hit)
    }

    fun getHistory(player: Player): List<HitData> = history[player.uniqueId]?.toList() ?: emptyList()

    fun removePlayer(player: Player) {
        history.remove(player.uniqueId)
    }
}