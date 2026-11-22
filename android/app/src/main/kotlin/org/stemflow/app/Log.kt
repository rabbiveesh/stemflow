package org.stemflow.app

import android.content.Context
import android.content.pm.ApplicationInfo
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.LinkedBlockingQueue

/**
 * android.util.Log, and in debug builds also a file of our own: logcat is a ring in
 * memory shared by every app, so on a busy phone our lines are gone within minutes.
 *
 * The file is capped: `logs/stemflow.log` up to [MAX_BYTES], then it becomes
 * `stemflow.1.log` (replacing the one before) and a new one starts. So at most
 * 2 × MAX_BYTES on disk. Read it with
 *   adb shell run-as org.stemflow.app cat files/logs/stemflow.1.log files/logs/stemflow.log
 *
 * The files in this package use this `Log` (they don't import android.util.Log).
 */
object Log {
    const val MAX_BYTES = 1L shl 20

    @Volatile private var dir: File? = null
    private val lines = LinkedBlockingQueue<String>(10_000)
    private val stamp = SimpleDateFormat("MM-dd HH:mm:ss.SSS", Locale.US)

    /** Start writing the file (debug builds only); safe to call more than once. */
    @Synchronized fun init(c: Context) {
        if (dir != null) return
        if (c.applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE == 0) return
        val d = File(c.filesDir, "logs").apply { mkdirs() }
        dir = d
        Thread({ writer(d) }, "stemflow-log").apply { isDaemon = true; priority = Thread.MIN_PRIORITY; start() }
        i(CaptureService.TAG, "log file: ${File(d, "stemflow.log").path}")
    }

    fun i(tag: String, msg: String): Int { keep('I', tag, msg); return android.util.Log.i(tag, msg) }
    fun w(tag: String, msg: String): Int { keep('W', tag, msg); return android.util.Log.w(tag, msg) }
    fun e(tag: String, msg: String): Int { keep('E', tag, msg); return android.util.Log.e(tag, msg) }
    fun e(tag: String, msg: String, t: Throwable): Int {
        keep('E', tag, msg + "\n" + android.util.Log.getStackTraceString(t))
        return android.util.Log.e(tag, msg, t)
    }

    private fun keep(level: Char, tag: String, msg: String) {
        if (dir == null) return
        val s = synchronized(stamp) { stamp.format(Date()) }
        // never block the audio threads: a full queue drops the line
        lines.offer("$s ${android.os.Process.myTid()} $level $tag: $msg\n")
    }

    private fun writer(d: File) {
        val f = File(d, "stemflow.log")
        var out: FileWriter? = null
        try {
            while (true) {
                val first = lines.take()
                if (out == null) out = FileWriter(f, true)
                out.write(first)
                // whatever else is waiting, in one go
                while (true) out.write(lines.poll() ?: break)
                out.flush()
                if (f.length() > MAX_BYTES) {
                    out.close()
                    out = null
                    f.renameTo(File(d, "stemflow.1.log"))
                }
            }
        } catch (e: Exception) {
            android.util.Log.e(CaptureService.TAG, "log file stopped: ${e.message}")
        } finally {
            try { out?.close() } catch (_: Exception) {}
        }
    }
}
