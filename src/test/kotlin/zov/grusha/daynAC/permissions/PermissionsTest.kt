package zov.grusha.daynAC.permissions

import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Права в [Permissions] и в plugin.yml должны совпадать: рассинхрон означает либо право,
 * которого нет в plugin.yml (default станет op/true в зависимости от сервера), либо
 * право в plugin.yml, которое код не проверяет.
 */
class PermissionsTest {

    private val pluginYml: String = File("src/main/resources/plugin.yml").readText()

    private val allPermissions = listOf(
        Permissions.ADMIN, Permissions.VIEW, Permissions.ALERTS, Permissions.SPECTATE,
        Permissions.RECORD, Permissions.TRAIN, Permissions.CONFIG,
        Permissions.EXEMPT, Permissions.EXEMPT_MANAGE
    )

    /**
     * Права, перечисленные в children у daynac.admin.
     *
     * Регулярка использует [\w.]+ (а не \w+), потому что имя может содержать
     * вторую точку: daynac.exempt.manage. Без этого тест молча пропустил бы
     * такое право при проверке «admin покрывает всё».
     */
    private fun adminChildren(): Set<String> {
        val block = pluginYml.substringAfter("\n  ${Permissions.ADMIN}:")
            .substringBefore("\n  ${Permissions.VIEW}:")
        return Regex("""^\s+(daynac\.[\w.]+):\s*true""", RegexOption.MULTILINE)
            .findAll(block).map { it.groupValues[1] }.toSet()
    }

    @Test
    fun `every permission constant is declared in plugin yml`() {
        for (permission in allPermissions) {
            val declared = Regex("^  ${Regex.escape(permission)}:", RegexOption.MULTILINE).containsMatchIn(pluginYml)
            assertTrue(declared, "$permission не объявлено в plugin.yml")
        }
    }

    /**
     * daynac.admin выдаётся OP-ам по умолчанию и включает все права, КРОМЕ
     * daynac.exempt: иначе OP не проверялся бы детекцией и на нём нельзя было
     * бы протестировать поведение фильтра. Временные исключения
     * (daynac.exempt.manage) в admin входят — это не иммунитет, а управление.
     */
    @Test
    fun `admin includes every permission except exempt`() {
        val expected = allPermissions.toSet() - Permissions.ADMIN - Permissions.EXEMPT
        assertEquals(expected, adminChildren())
    }

    @Test
    fun `exempt is never granted by admin`() {
        assertFalse(Permissions.EXEMPT in adminChildren())
    }

    @Test
    fun `every subcommand permission is a known permission`() {
        for ((sub, permission) in Permissions.SUBCOMMANDS) {
            assertTrue(permission in allPermissions, "$sub требует неизвестное право $permission")
        }
    }

    @Test
    fun `dataset and training permissions are separate from view`() {
        assertEquals(Permissions.RECORD, Permissions.SUBCOMMANDS["record"])
        assertEquals(Permissions.TRAIN, Permissions.SUBCOMMANDS["train"])
        assertEquals(Permissions.TRAIN, Permissions.SUBCOMMANDS["eval"])
        assertFalse(Permissions.SUBCOMMANDS["record"] == Permissions.VIEW)
    }

    @Test
    fun `clearflags and exempt subcommands are wired to expected permissions`() {
        assertEquals(Permissions.CONFIG, Permissions.SUBCOMMANDS["clearflags"])
        assertEquals(Permissions.EXEMPT_MANAGE, Permissions.SUBCOMMANDS["exempt"])
    }
}