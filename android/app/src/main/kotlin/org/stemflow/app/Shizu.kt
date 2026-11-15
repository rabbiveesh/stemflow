package org.stemflow.app

import android.content.ComponentName
import android.content.Context
import android.content.ServiceConnection
import android.content.pm.PackageManager
import android.os.IBinder
import android.os.Parcel
import android.util.Log
import rikka.shizuku.Shizuku

/**
 * Starts the capture helper through Shizuku, so no computer is needed. Shizuku is a
 * separate app the user installs and starts (over wireless debugging, once per
 * boot); it runs HelperService as the shell uid for us.
 */
object Shizu {
    private var conn: ServiceConnection? = null
    private var args: Shizuku.UserServiceArgs? = null

    /** Shizuku is installed and its service is running. */
    fun available() = try { Shizuku.pingBinder() } catch (_: Throwable) { false }

    fun granted() = available() && try {
        Shizuku.checkSelfPermission() == PackageManager.PERMISSION_GRANTED
    } catch (_: Throwable) { false }

    fun state() = when {
        granted() -> "Shizuku ready"
        available() -> "Shizuku needs permission (switch on to ask)"
        else -> "Shizuku not running (or run android/helper.sh)"
    }

    /** Ask the user, through Shizuku's own dialog; `then` gets the answer. */
    fun request(then: (Boolean) -> Unit) {
        val l = object : Shizuku.OnRequestPermissionResultListener {
            override fun onRequestPermissionResult(code: Int, result: Int) {
                Shizuku.removeRequestPermissionResultListener(this)
                then(result == PackageManager.PERMISSION_GRANTED)
            }
        }
        Shizuku.addRequestPermissionResultListener(l)
        Shizuku.requestPermission(1)
    }

    /** Start the helper: all media playback except `exclude`'s, into our socket on `port`. */
    fun start(c: Context, port: Int, exclude: IntArray) {
        val a = Shizuku.UserServiceArgs(ComponentName(c.packageName, HelperService::class.java.name))
            .daemon(false)
            .processNameSuffix("helper")
            .debuggable(true)
            .version(1)
        val sc = object : ServiceConnection {
            override fun onServiceConnected(name: ComponentName, binder: IBinder) {
                val data = Parcel.obtain()
                val reply = Parcel.obtain()
                try {
                    data.writeInt(port)
                    data.writeIntArray(exclude)
                    binder.transact(1, data, reply, 0) // HelperService.START
                    reply.readException()
                    Log.i(CaptureService.TAG, "helper started through Shizuku")
                } catch (e: Throwable) {
                    Log.e(CaptureService.TAG, "starting the helper through Shizuku failed", e)
                } finally {
                    data.recycle()
                    reply.recycle()
                }
            }
            override fun onServiceDisconnected(name: ComponentName) {}
        }
        args = a
        conn = sc
        Shizuku.bindUserService(a, sc)
    }

    fun stop() {
        val a = args ?: return
        try { Shizuku.unbindUserService(a, conn, true) } catch (_: Throwable) {}
        args = null
        conn = null
    }
}
