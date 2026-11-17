package org.stemflow.app;

import android.content.Context;
import android.os.Binder;
import android.os.Parcel;

/**
 * Helper as a Shizuku "user service": Shizuku starts this class in a process of its
 * own running as the shell uid and hands the app its binder. One call, START, with
 * the app's port, the uids to capture (empty: all media) and the uids to leave out; the process exits when the app closes the
 * socket or unbinds.
 */
public class HelperService extends Binder {
    static final int START = 1;
    /** Bring the app back to the front (the shell may start activities; we, from the
     * background, may not): after recall opened the player to switch tracks. */
    static final int FRONT = 2;
    /** Shizuku's own "destroy" transaction (IBinder.LAST_CALL_TRANSACTION - 100). */
    private static final int DESTROY = 16777115;

    public HelperService() {}

    public HelperService(Context context) {}

    @Override
    protected boolean onTransact(int code, Parcel data, Parcel reply, int flags) {
        if (code == DESTROY) {
            System.exit(0);
            return true;
        }
        if (code == FRONT) {
            final String component = data.readString();
            try {
                new ProcessBuilder("am", "start", "-n", component, "--activity-reorder-to-front")
                    .redirectErrorStream(true).start().waitFor();
                if (reply != null) reply.writeNoException();
            } catch (Exception e) {
                if (reply != null) reply.writeException(new IllegalStateException(e.toString()));
            }
            return true;
        }
        if (code == START) {
            final int port = data.readInt();
            final int[] only = data.createIntArray();
            final int[] exclude = data.createIntArray();
            new Thread(() -> {
                try {
                    Helper.run(port, only, exclude);
                } catch (Throwable t) {
                    android.util.Log.e("stemflow", "helper failed", t);
                    System.exit(1);
                }
            }, "stemflow-helper").start();
            if (reply != null) reply.writeNoException();
            return true;
        }
        try {
            return super.onTransact(code, data, reply, flags);
        } catch (android.os.RemoteException e) {
            return false;
        }
    }
}
