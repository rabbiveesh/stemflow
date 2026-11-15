package org.stemflow.app

import android.content.Context
import android.content.Intent

/**
 * What the capture helper takes: everything that is playing (the default), or one
 * chosen player. The choices are the installed apps that offer a media service
 * (music and podcast players), found through the manifest's <queries>.
 */
object Sources {
    class App(val pkg: String, val label: String, val uid: Int)

    private val actions = listOf(
        "android.media.browse.MediaBrowserService",
        "androidx.media3.session.MediaSessionService",
        "androidx.media3.session.MediaLibraryService",
    )

    fun apps(c: Context): List<App> {
        val pm = c.packageManager
        return actions
            .flatMap { pm.queryIntentServices(Intent(it), 0) }
            .map { it.serviceInfo.applicationInfo }
            .filter { it.packageName != c.packageName }
            .distinctBy { it.packageName }
            .map { App(it.packageName, pm.getApplicationLabel(it).toString(), it.uid) }
            .sortedBy { it.label.lowercase() }
    }

    private fun prefs(c: Context) = c.getSharedPreferences("models", Context.MODE_PRIVATE)

    /** The chosen player, or null for everything (also when the choice was uninstalled). */
    fun current(c: Context): App? {
        val pkg = prefs(c).getString("source", null) ?: return null
        return apps(c).firstOrNull { it.pkg == pkg }
    }

    fun select(c: Context, app: App?) = prefs(c).edit().putString("source", app?.pkg).apply()
}
