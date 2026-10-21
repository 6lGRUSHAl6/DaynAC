plugins {
    kotlin("jvm") version "2.4.20"
    id("com.gradleup.shadow") version "9.6.1"
    id("xyz.jpenilla.run-paper") version "3.1.0"
}

repositories {
    mavenCentral()
    maven("https://repo.papermc.io/repository/maven-public/")
}

// Версия Minecraft для `gradle runServer`. На сборку плагина НЕ влияет: плагин всегда собирается
// против самого старого поддерживаемого API (1.21.11, Java 21), а на более новых серверах
// (26.1, 26.2, ...) работает тем же jar. Проверка на другой версии:
//   gradle runServer -PmcVersion=26.1.2
val testMcVersion = providers.gradleProperty("mcVersion").getOrElse("1.21.11")
// Paper 26.1+ требует Java 25 на сервере, 1.21.x — Java 21
val testServerJava = if ((testMcVersion.substringBefore('.').toIntOrNull() ?: 0) >= 26) 25 else 21
// Launcher считается здесь, на верхнем уровне скрипта: внутри tasks { runServer { ... } }
// `javaToolchains.launcherFor` не разрешается (Unresolved reference). Это Provider, JDK ищется
// только при реальном запуске runServer — для обычной сборки JDK 25 не нужна.
val serverJavaLauncher = extensions.getByType<JavaToolchainService>().launcherFor {
    languageVersion.set(JavaLanguageVersion.of(testServerJava))
}

dependencies {
    compileOnly("io.papermc.paper:paper-api:1.21.11-R0.1-SNAPSHOT")
    implementation(kotlin("stdlib-jdk8"))

    //тесты
    testImplementation(kotlin("test-junit5"))
    testImplementation("org.junit.jupiter:junit-jupiter:5.11.4")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher:1.11.4") // Явно указываем версию
    // paper-api объявлен compileOnly и в тестовый classpath не попадает: без этого тесты
    // не видят org.bukkit.* (Location, Player, YamlConfiguration, ItemStack...).
    testImplementation("io.papermc.paper:paper-api:1.21.11-R0.1-SNAPSHOT")
    // Моки Bukkit-объектов (Player, JavaPlugin) и статики Bukkit.getScheduler/getOnlinePlayers.
    // Mockito 5 использует inline mockmaker по умолчанию, поэтому мокает и final-классы Kotlin.
    testImplementation("org.mockito:mockito-core:5.18.0")
    testImplementation("org.mockito.kotlin:mockito-kotlin:5.4.0")
    // Полноценный mock-сервер: нужен только GUI-тестам (реестр Material, ItemMeta, PDC, инвентари).
    testImplementation("org.mockbukkit.mockbukkit:mockbukkit-v1.21:4.116.3")
}

kotlin {
    jvmToolchain(21)
}

tasks {
    build {
        dependsOn(shadowJar)
    }

    test {
        useJUnitPlatform()
    }

    runServer {
        // Configure the Minecraft version for our task.
        // This is the only required configuration besides applying the plugin.
        // Your plugin's jar (or shadowJar if present) will be used automatically.
        minecraftVersion(testMcVersion)
        javaLauncher.set(serverJavaLauncher)
        jvmArgs("-Xms2G", "-Xmx2G")
    }

    processResources {
        val props = mapOf("version" to version)
        filesMatching("plugin.yml") {
            expand(props)
        }
    }
}