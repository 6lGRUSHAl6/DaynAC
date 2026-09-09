package zov.grusha.daynAC.tracking

import org.bukkit.entity.Player
import java.util.ArrayDeque
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap

class HitTracker(private val maxSize: Int = 8) {
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