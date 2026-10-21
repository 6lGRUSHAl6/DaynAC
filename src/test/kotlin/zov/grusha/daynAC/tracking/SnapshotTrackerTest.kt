package zov.grusha.daynAC.tracking

import org.bukkit.Location
import org.bukkit.entity.Player
import org.bukkit.util.BoundingBox
import org.junit.jupiter.api.BeforeEach
import org.mockito.kotlin.mock
import org.mockito.kotlin.whenever
import java.util.UUID
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.log2
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * SnapshotTracker: агрегаты по истории поворотов, тайминги ударов и сборка HitData.
 *
 * Player — Mockito-мок, но Location настоящая: `player.location` отвечает текущим
 * объектом из [locationRef], поэтому тест сам двигает игрока между снимками.
 * Время подменяется через инъекцию [SnapshotTracker.clock] — без Thread.sleep.
 */
class SnapshotTrackerTest {

    private val locationRef = AtomicReference(Location(null, 0.0, 64.0, 0.0, 0f, 0f))
    private var now = 0L
    private lateinit var uuid: UUID
    private lateinit var player: Player
    private lateinit var tracker: SnapshotTracker

    @BeforeEach
    fun setUp() {
        uuid = UUID.randomUUID()
        player = mock()
        whenever(player.uniqueId).thenReturn(uuid)
        whenever(player.location).thenAnswer { locationRef.get() }
        tracker = SnapshotTracker(clock = { now })
    }

    /** Ставит игрока в точку; вызывать между снимками. */
    private fun place(yaw: Float = 0f, pitch: Float = 0f, x: Double = 0.0, z: Double = 0.0) {
        locationRef.set(Location(null, x, 64.0, z, yaw, pitch))
    }

    /** Делает снимок на текущей позиции. */
    private fun snapshot(yaw: Float = 0f, pitch: Float = 0f, x: Double = 0.0, z: Double = 0.0): PlayerSnapshot? {
        place(yaw, pitch, x, z)
        return tracker.recordSnapshot(player)
    }

    private fun hit(currentAimAngle: Double? = null): Pair<Long?, Double?> =
        tracker.registerHitWithAngle(player, currentAimAngle)

    // --- recordSnapshot: дельты, wrap, скорость -------------------------

    @Test
    fun `first snapshot only initializes state`() {
        assertNull(snapshot())
        assertTrue(tracker.getSnapshot(player).isEmpty())
    }

    @Test
    fun `snapshot computes rotation and movement deltas`() {
        snapshot(yaw = 0f, pitch = 0f)
        val snap = snapshot(yaw = 90f, pitch = 10f, x = 3.0, z = 4.0)!!

        assertEquals(90f, snap.yawDeltaAbs, 1e-4f)
        assertEquals(10f, snap.pitchDeltaAbs, 1e-4f)
        assertEquals(90f, snap.yawDeltaSigned, 1e-4f)
        assertEquals(10f, snap.pitchDeltaSigned, 1e-4f)
        // Первый снимок: предыдущая дельта считалась нулевой.
        assertEquals(90f, snap.yawAccel, 1e-4f)
        assertEquals(10f, snap.pitchAccel, 1e-4f)
        assertEquals(5f, snap.speedXZ, 1e-4f)
    }

    @Test
    fun `yaw wrap uses shortest signed path`() {
        snapshot(yaw = 350f)
        val forward = snapshot(yaw = 5f)!!
        assertEquals(15f, forward.yawDeltaSigned, 1e-4f)

        snapshot(yaw = 5f)
        val backward = snapshot(yaw = 350f)!!
        assertEquals(-15f, backward.yawDeltaSigned, 1e-4f)
    }

    @Test
    fun `rotation across 180 degrees picks negative direction`() {
        snapshot(yaw = 0f)
        val snap = snapshot(yaw = 270f)!!
        assertEquals(-90f, snap.yawDeltaSigned, 1e-4f)
        assertEquals(90f, snap.yawDeltaAbs, 1e-4f)
    }

    @Test
    fun `acceleration is absolute change of rotation speed`() {
        snapshot(yaw = 0f)
        snapshot(yaw = 10f)               // delta 10, accel 10
        val second = snapshot(yaw = 20f)!! // delta 10, accel 0
        val third = snapshot(yaw = 40f)!!  // delta 20, accel 10

        assertEquals(0f, second.yawAccel, 1e-4f)
        assertEquals(10f, third.yawAccel, 1e-4f)
    }

    @Test
    fun `teleport-like movement clamps speed to zero`() {
        snapshot(x = 0.0, z = 0.0)
        val snap = snapshot(x = 50.0, z = 0.0)!!
        assertEquals(0f, snap.speedXZ, 1e-4f)
    }

    @Test
    fun `history is capped at maxSize`() {
        tracker = SnapshotTracker(maxSize = 3, clock = { now })
        snapshot(yaw = 0f)
        snapshot(yaw = 1f)
        snapshot(yaw = 2f)
        snapshot(yaw = 3f)
        val history = tracker.getSnapshot(player)

        assertEquals(3, history.size)
        assertEquals(1f, history[0].yawDeltaAbs, 1e-4f)
        assertEquals(1f, history[1].yawDeltaAbs, 1e-4f)
        assertEquals(1f, history[2].yawDeltaAbs, 1e-4f)
    }

    // --- Энтропии ---------------------------------------------------------

    @Test
    fun `yaw entropy is log2 of bin count for uniform distribution`() {
        snapshot(yaw = 0f)
        var yaw = 0f
        // Дельты 1,3,5,...,19 -> ровно по одному попаданию в каждый из 10 бинов [0,20).
        for (delta in 1..19 step 2) {
            yaw += delta
            snapshot(yaw = yaw)
        }

        assertEquals(log2(10.0), tracker.getYawEntropyAbs(player)!!, 1e-9)
    }

    @Test
    fun `entropy is zero when every rotation is identical`() {
        snapshot(yaw = 0f)
        var yaw = 0f
        repeat(10) {
            yaw += 5f
            snapshot(yaw = yaw)
        }

        assertEquals(0.0, tracker.getYawEntropyAbs(player)!!, 1e-9)
    }

    @Test
    fun `large rotations are clamped into the last bin`() {
        snapshot(yaw = 0f)
        var yaw = 0f
        repeat(10) {
            yaw += 100f
            snapshot(yaw = yaw)
        }

        assertEquals(0.0, tracker.getYawEntropyAbs(player)!!, 1e-9)
        assertEquals(0.0, tracker.getYawEntropySigned(player)!!, 1e-9)
    }

    @Test
    fun `signed entropy uses 11 bins over plus minus 20 degrees`() {
        // Центры всех 11 бинов диапазона [-20, +20].
        val binWidth = 40.0 / 11.0
        val deltas = (0 until 11).map { -20.0 + (it + 0.5) * binWidth }

        snapshot(yaw = 0f)
        var yaw = 0f
        for (delta in deltas) {
            yaw += delta.toFloat()
            snapshot(yaw = yaw)
        }

        assertEquals(log2(11.0), tracker.getYawEntropySigned(player)!!, 1e-9)
    }

    @Test
    fun `pitch entropy uses its own plus minus 10 degree range`() {
        snapshot(pitch = 0f)
        var pitch = 0f
        repeat(10) {
            pitch += 100f
            snapshot(pitch = pitch)
        }
        // 100 градусов за пределами ±10 -> всё в крайнем бине.
        assertEquals(0.0, tracker.getPitchEntropyAbs(player)!!, 1e-9)
        assertEquals(0.0, tracker.getPitchEntropySigned(player)!!, 1e-9)
    }

    // --- Джиттер и плавность ---------------------------------------------

    @Test
    fun `jitter is population standard deviation of deltas`() {
        snapshot(yaw = 0f)
        snapshot(yaw = 10f)
        snapshot(yaw = 30f)
        snapshot(yaw = 60f) // дельты 10, 20, 30

        val expected = kotlin.math.sqrt(200.0 / 3.0) // mean 20, сумма кв. отклонений 200
        assertEquals(expected, tracker.getYawJitterAbs(player)!!, 1e-9)
        assertEquals(expected, tracker.getYawJitterSigned(player)!!, 1e-9)
    }

    @Test
    fun `jitter is zero for constant rotation`() {
        snapshot(yaw = 0f)
        var yaw = 0f
        repeat(5) {
            yaw += 10f
            snapshot(yaw = yaw)
        }
        assertEquals(0.0, tracker.getYawJitterAbs(player)!!, 1e-9)
    }

    @Test
    fun `smoothness is max to average ratio`() {
        snapshot(yaw = 0f)
        snapshot(yaw = 10f)
        snapshot(yaw = 30f)
        snapshot(yaw = 60f) // дельты 10, 20, 30

        assertEquals(1.5, tracker.getRotationSmoothnessYaw(player)!!, 1e-9)
    }

    @Test
    fun `smoothness is null when there is no rotation`() {
        snapshot(yaw = 0f)
        snapshot(yaw = 0f)
        snapshot(yaw = 0f)
        assertNull(tracker.getRotationSmoothnessYaw(player))
        assertNull(tracker.getRotationSmoothnessPitch(player))
    }

    @Test
    fun `micro adjustment counts only tiny non-zero deltas`() {
        snapshot(yaw = 0f)
        // Дельты: 2^-10 (слишком мало), 0.25, 0.5 (граница), 0.125, 0.
        // Значения — двоичные дроби, поэтому Float считает их без накопленной ошибки.
        val yaws = listOf(0.0009765625f, 0.2509765625f, 0.7509765625f, 0.8759765625f, 0.8759765625f)
        for (yaw in yaws) snapshot(yaw = yaw)

        assertEquals(2, tracker.getMicroAdjustCountYaw(player))
    }

    // --- Jerk и прямолинейность ------------------------------------------

    @Test
    fun `jerk averages changes of acceleration across yaw and pitch`() {
        snapshot(yaw = 0f)
        snapshot(yaw = 10f)
        snapshot(yaw = 20f)
        snapshot(yaw = 40f) // ускорения yaw: 10, 0, 10

        // Два перехода: |0-10|/2 = 5 и |10-0|/2 = 5.
        assertEquals(5.0, tracker.getJerkValue(player)!!, 1e-9)
    }

    @Test
    fun `jerk is zero for uniform rotation`() {
        // maxSize = 3 отбрасывает самый первый снимок: у него yawAccel = delta
        // из-за нулевой предыдущей дельты, и этот стартовый переход — не «джерк».
        tracker = SnapshotTracker(maxSize = 3, clock = { now })
        snapshot(yaw = 0f)
        var yaw = 0f
        repeat(5) {
            yaw += 10f
            snapshot(yaw = yaw)
        }
        assertEquals(0.0, tracker.getJerkValue(player)!!, 1e-9)
    }

    @Test
    fun `straight line ratio is 1 for constant angular velocity`() {
        snapshot(yaw = 0f)
        var yaw = 0f
        repeat(6) {
            yaw += 10f
            snapshot(yaw = yaw)
        }

        assertEquals(1.0, tracker.getStraightLineRatio(player)!!, 1e-9)
    }

    @Test
    fun `straight line ratio drops for irregular rotation`() {
        snapshot(yaw = 0f)
        snapshot(yaw = 10f)
        snapshot(yaw = 5f)   // signed дельта -5
        snapshot(yaw = 25f)  // +20
        snapshot(yaw = 22f)  // -3
        snapshot(yaw = 42f)  // +20

        val r2 = tracker.getStraightLineRatio(player)!!
        assertTrue(r2 in 0.0..1.0)
        assertTrue(r2 < 0.99, "неровный поворот не должен выглядеть линейным, R2=$r2")
    }

    @Test
    fun `straight line ratio is null when cumulative rotation is flat`() {
        snapshot(yaw = 0f)
        snapshot(yaw = 0f)
        snapshot(yaw = 0f)
        snapshot(yaw = 0f)
        assertNull(tracker.getStraightLineRatio(player))
    }

    // --- getHitTimeCV и тайминги -----------------------------------------

    @Test
    fun `hit time CV is null before enough hits`() {
        assertNull(tracker.getHitTimeCV(player))

        now = 0
        hit()
        assertNull(tracker.getHitTimeCV(player), "первый удар не даёт интервала")

        now = 100
        hit()
        assertNull(tracker.getHitTimeCV(player), "одного интервала мало")
    }

    @Test
    fun `hit time CV matches known sample`() {
        now = 0; hit()
        now = 100; hit()
        now = 300; hit()
        now = 600; hit() // интервалы 100, 200, 300

        // mean 200, выборочное std 100 -> CV 0.5
        assertEquals(0.5, tracker.getHitTimeCV(player)!!, 1e-9)
    }

    @Test
    fun `hit time CV is zero for metronomic timing`() {
        now = 0; hit()
        repeat(5) {
            now += 100
            hit()
        }
        assertEquals(0.0, tracker.getHitTimeCV(player)!!, 1e-9)
    }

    @Test
    fun `pause longer than two seconds resets the hit series`() {
        now = 0; hit()
        now = 100; hit()
        now = 200; hit()
        assertEquals(0.0, tracker.getHitTimeCV(player)!!, 1e-9)

        now = 2_200; hit()          // пауза 2000 мс — серия очищена
        assertNull(tracker.getHitTimeCV(player))

        now = 2_300; hit()          // первый интервал новой серии
        assertNull(tracker.getHitTimeCV(player))

        now = 2_400; hit()
        assertEquals(0.0, tracker.getHitTimeCV(player)!!, 1e-9)
    }

    @Test
    fun `hit time CV keeps only the last ten intervals`() {
        now = 0; hit()
        repeat(10) {                 // 10 интервалов по 100 мс
            now += 100
            hit()
        }
        now += 200; hit()            // 11-й интервал 200 мс, самый старый вытеснен

        // Последние 10: девять по 100 и один 200.
        // mean 110, выборочная дисперсия 1000 -> std 31.6228 -> CV 0.28748.
        val expected = kotlin.math.sqrt(1000.0) / 110.0
        assertEquals(expected, tracker.getHitTimeCV(player)!!, 1e-9)
    }

    @Test
    fun `registerHitWithAngle returns previous aim angle and stores the new one`() {
        now = 0
        assertEquals(Pair<Long?, Double?>(null, null), hit(5.0))
        now = 100
        assertEquals(Pair<Long?, Double?>(100L, 5.0), hit(7.0))
        now = 200
        assertEquals(Pair<Long?, Double?>(100L, 7.0), hit(null))
    }

    @Test
    fun `invalid aim angles are not stored`() {
        hit(5.0)

        hit(-1.0)
        assertEquals(5.0, hit(9.0).second)

        hit(Double.NaN)
        assertEquals(9.0, hit(11.0).second)

        hit(Double.POSITIVE_INFINITY)
        assertEquals(11.0, hit(13.0).second)

        hit(null)
        assertEquals(13.0, hit(15.0).second)
    }

    @Test
    fun `deprecated registerHit still returns hit delta`() {
        @Suppress("DEPRECATION")
        assertNull(tracker.registerHit(player))

        now = 150
        @Suppress("DEPRECATION")
        assertEquals(150L, tracker.registerHit(player))
    }

    @Test
    fun `resetHitTiming clears intervals but keeps rotation snapshots`() {
        now = 0; hit()
        now = 100; hit()
        now = 200; hit()
        snapshot(yaw = 10f)
        snapshot(yaw = 20f)
        assertEquals(0.0, tracker.getHitTimeCV(player)!!, 1e-9)

        tracker.resetHitTiming(uuid)

        assertNull(tracker.getHitTimeCV(player))
        assertTrue(tracker.getSnapshot(player).isNotEmpty(), "снимки поворотов не трогаются")

        now += 100
        assertNull(hit().first, "первый удар после сброса не даёт интервала")
    }

    @Test
    fun `resetHitTiming for unknown player is a no-op`() {
        tracker.resetHitTiming(UUID.randomUUID())
        assertEquals(Pair<Long?, Double?>(null, null), hit())
    }

    // --- Скорость и очистка ----------------------------------------------

    @Test
    fun `speed is null before the first snapshot`() {
        assertNull(tracker.getSpeedXZ(player))
    }

    @Test
    fun `speed reflects the latest snapshot`() {
        snapshot(x = 0.0, z = 0.0)
        snapshot(x = 0.0, z = 3.0)
        assertEquals(3f, tracker.getSpeedXZ(player)!!, 1e-4f)
    }

    @Test
    fun `removePlayer clears snapshots and aim angle history`() {
        snapshot(yaw = 0f)
        snapshot(yaw = 10f)
        hit(5.0)
        hit(7.0)
        assertEquals(1, tracker.getSnapshot(player).size)

        tracker.removePlayer(player)

        assertTrue(tracker.getSnapshot(player).isEmpty())
        assertNull(tracker.getYawJitterAbs(player))
        assertEquals(Pair<Long?, Double?>(null, null), hit(), "состояние игрока создано заново")
    }

    @Test
    fun `players do not share state`() {
        val otherRef = AtomicReference(Location(null, 0.0, 64.0, 0.0, 0f, 0f))
        val other: Player = mock()
        whenever(other.uniqueId).thenReturn(UUID.randomUUID())
        whenever(other.location).thenAnswer { otherRef.get() }

        snapshot(yaw = 0f)
        snapshot(yaw = 10f)
        snapshot(yaw = 20f)

        tracker.recordSnapshot(other)

        assertEquals(2, tracker.getSnapshot(player).size)
        assertEquals(0, tracker.getSnapshot(other).size)
        assertNull(tracker.getYawJitterAbs(other))
        assertNotNull(tracker.getYawJitterAbs(player))
    }

    // --- buildHitData ------------------------------------------------------

    @Test
    fun `buildHitData assembles aim, reach and timing fields`() {
        val world: org.bukkit.World = mock()
        whenever(player.world).thenReturn(world)
        val eye = Location(world, 0.0, 65.0, 0.0, 0f, 0f)
        whenever(player.eyeLocation).thenReturn(eye)

        val victim: Player = mock()
        whenever(victim.uniqueId).thenReturn(UUID.randomUUID())
        whenever(victim.world).thenReturn(world)
        // Хитбокс в трёх блоках впереди по +Z.
        whenever(victim.boundingBox).thenReturn(BoundingBox(-0.3, 64.0, 2.7, 0.3, 65.8, 3.3))

        now = 1_000
        val first = tracker.buildHitData(player, victim)

        assertNull(first.hitTimeDelta)
        assertNull(first.snapFactor, "первый удар не имеет предыдущего угла")
        assertEquals(1_000L, first.timestamp)
        assertNotNull(first.aimAngle)
        assertTrue(first.aimAngle < 5.0, "игрок смотрит почти в центр тела, angle=${first.aimAngle}")
        assertTrue(first.reachDistance > 0.0)
        assertTrue(first.distance > 0.0)
        assertNull(first.hitTimeCV)
        assertNull(first.yawJitterAbs)

        now = 1_250
        val second = tracker.buildHitData(player, victim)

        assertEquals(250L, second.hitTimeDelta)
        assertNotNull(second.snapFactor)
    }
}
