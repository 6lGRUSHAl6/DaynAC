package zov.grusha.daynAC.exemptions

import org.bukkit.configuration.file.YamlConfiguration
import org.junit.jupiter.api.io.TempDir
import java.io.File
import java.util.UUID
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * ExemptionManager — файловое хранилище временных исключений.
 * Bukkit-сервер для тестов не нужен: YamlConfiguration работает headless,
 * а dataFolder подставляется через JUnit @TempDir.
 */
class ExemptionManagerTest {

    @TempDir
    lateinit var tempDir: File

    private fun manager() = ExemptionManager(tempDir)

    private fun exemptionsFile() = File(tempDir, "exemptions.yml")

    private fun player(): UUID = UUID.randomUUID()

    // --- Выдача исключений ----------------------------------------------

    @Test
    fun `permanent exemption is visible and reported as PERMANENT`() {
        val manager = manager()
        val id = player()

        val expiry = manager.exempt(id, 0)

        assertEquals(ExemptionManager.PERMANENT, expiry)
        assertTrue(manager.isExempt(id))
        assertEquals(ExemptionManager.PERMANENT, manager.getExpiry(id))
        assertEquals(1, manager.size())
    }

    @Test
    fun `negative duration also means permanent`() {
        val manager = manager()
        val id = player()

        assertEquals(ExemptionManager.PERMANENT, manager.exempt(id, -5000))
        assertTrue(manager.isExempt(id))
    }

    @Test
    fun `temporary exemption expires at now plus duration`() {
        val manager = manager()
        val id = player()

        val before = System.currentTimeMillis()
        val expiry = manager.exempt(id, 60_000)
        val after = System.currentTimeMillis()

        assertTrue(expiry in (before + 60_000)..(after + 60_000), "expiry=$expiry")
        assertTrue(manager.isExempt(id))
    }

    @Test
    fun `unknown player is not exempt`() {
        val manager = manager()

        assertFalse(manager.isExempt(player()))
        assertNull(manager.getExpiry(UUID.randomUUID()))
        assertEquals(0, manager.size())
    }

    // --- Снятие и истечение ---------------------------------------------

    @Test
    fun `unexempt removes record and returns whether it existed`() {
        val manager = manager()
        val id = player()
        manager.exempt(id, 0)

        assertTrue(manager.unexempt(id))
        assertFalse(manager.isExempt(id))
        assertNull(manager.getExpiry(id))
        assertFalse(manager.unexempt(id), "повторное снятие ничего не удаляет")
    }

    @Test
    fun `isExempt lazily removes an expired record`() {
        val id = player()
        // Пишем файл вручную: срок истёк 60 секунд назад.
        writeFile(mapOf(id to System.currentTimeMillis() - 60_000))
        val manager = manager()

        // load() не фильтрует просроченные — запись ещё в карте.
        assertEquals(1, manager.size())
        assertFalse(manager.isExempt(id))
        assertEquals(0, manager.size(), "просроченная запись удалена при проверке")
    }

    @Test
    fun `cleanupExpired removes only expired records`() {
        val expired = player()
        val permanent = player()
        val future = player()
        writeFile(
            mapOf(
                expired to System.currentTimeMillis() - 1_000,
                permanent to ExemptionManager.PERMANENT,
                future to System.currentTimeMillis() + 60_000
            )
        )
        val manager = manager()

        manager.cleanupExpired()

        assertFalse(manager.isExempt(expired))
        assertTrue(manager.isExempt(permanent))
        assertTrue(manager.isExempt(future))
        assertEquals(2, manager.size())
    }

    @Test
    fun `cleanupExpired persists the cleaned list`() {
        val expired = player()
        val permanent = player()
        writeFile(mapOf(expired to System.currentTimeMillis() - 1_000, permanent to ExemptionManager.PERMANENT))
        val manager = manager()

        manager.cleanupExpired()

        val reloaded = YamlConfiguration.loadConfiguration(exemptionsFile())
            .getConfigurationSection("exemptions")!!
        assertEquals(setOf(permanent.toString()), reloaded.getKeys(false))
    }

    // --- Персистентность -------------------------------------------------

    @Test
    fun `exemptions survive a manager restart`() {
        val permanent = player()
        val temporary = player()
        manager().apply {
            exempt(permanent, 0)
            exempt(temporary, 120_000)
        }

        val restored = manager()

        assertEquals(2, restored.size())
        assertEquals(ExemptionManager.PERMANENT, restored.getExpiry(permanent))
        assertTrue(restored.isExempt(permanent))
        assertTrue(restored.isExempt(temporary))
    }

    @Test
    fun `unexempt persists removal`() {
        val id = player()
        manager().exempt(id, 0)

        manager().unexempt(id)

        val restored = manager()
        assertEquals(0, restored.size())
        assertFalse(restored.isExempt(id))
    }

    @Test
    fun `temporary expiry value survives a restart unchanged`() {
        val id = player()
        val expiry = manager().exempt(id, 3_600_000)

        assertEquals(expiry, manager().getExpiry(id))
    }

    @Test
    fun `missing file is treated as empty list`() {
        assertFalse(exemptionsFile().exists())
        val manager = manager()

        assertEquals(0, manager.size())
        assertFalse(manager.isExempt(player()))
    }

    // --- Битые данные ------------------------------------------------------

    @Test
    fun `invalid uuid in file is skipped without crashing`() {
        writeFileRaw(
            mapOf(
                "not-a-uuid" to 123L,
                "also bad" to 456L,
                player().toString() to ExemptionManager.PERMANENT
            )
        )

        val manager = manager()

        assertEquals(1, manager.size(), "валидная запись загружена, битые пропущены")
    }

    @Test
    fun `non-positive expiry in file is loaded as permanent`() {
        val id = player()
        writeFile(mapOf(id to -42L))

        val manager = manager()

        assertEquals(ExemptionManager.PERMANENT, manager.getExpiry(id))
        assertTrue(manager.isExempt(id))
    }

    @Test
    fun `empty file without exemptions section is tolerated`() {
        exemptionsFile().writeText("other-section: true\n")

        val manager = manager()

        assertEquals(0, manager.size())
    }

    // --- helpers -----------------------------------------------------------

    private fun writeFile(entries: Map<UUID, Long>) {
        writeFileRaw(entries.mapKeys { it.key.toString() })
    }

    private fun writeFileRaw(entries: Map<String, Long>) {
        val yaml = YamlConfiguration()
        for ((key, value) in entries) {
            yaml.set("exemptions.$key", value)
        }
        yaml.save(exemptionsFile())
    }
}
