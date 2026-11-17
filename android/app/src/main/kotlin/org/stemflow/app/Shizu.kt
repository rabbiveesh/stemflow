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
    @Volatile private var binder: IBinder? = null

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

    /**
     * Start the helper, streaming into our socket on `port`: just `only`'s playback,
     * or with `only` empty all media playback except `exclude`'s.
     */
    fun start(c: Context, port: Int, only: IntArray, exclude: IntArray) {
        val a = Shizuku.UserServiceArgs(ComponentName(c.packageName, HelperService::class.java.name))
            .daemon(false)
            .processNameSuffix("helper")
            .debuggable(true)
            .version(3)
        val sc = object : ServiceConnection {
            override fun onServiceConnected(name: ComponentName, binder: IBinder) {
                this@Shizu.binder = binder
                val data = Parcel.obtain()
                val reply = Parcel.obtain()
                try {
                    data.writeInt(port)
                    data.writeIntArray(only)
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
            override fun onServiceDisconnected(name: ComponentName) { this@Shizu.binder = null }
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
        binder = null
    }

    /** Bring stemflow to the front through the helper; false if it isn't running. */
    fun front(c: Context): Boolean {
        val b = binder ?: return false
        val data = Parcel.obtain()
        val reply = Parcel.obtain()
        return try {
            data.writeString("${c.packageName}/${MainActivity::class.java.name}")
            b.transact(2, data, reply, 0) // HelperService.FRONT
            reply.readException()
            true
        } catch (e: Throwable) {
            Log.e(CaptureService.TAG, "bringing stemflow back failed", e)
            false
        } finally {
            data.recycle()
            reply.recycle()
        }
    }
}
