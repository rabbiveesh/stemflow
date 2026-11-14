package org.stemflow.app;

import android.annotation.SuppressLint;
import android.app.Application;
import android.content.AttributionSource;
import android.content.Context;
import android.content.ContextWrapper;
import android.content.pm.ApplicationInfo;
import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.AudioRecord;
import android.os.Looper;
import android.os.Process;

import java.io.OutputStream;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.net.InetAddress;
import java.net.Socket;

/**
 * Captures chosen apps' playback with shell privileges and streams it to the app.
 *
 * Spotify marks its audio NO_MEDIA_PROJECTION, so an ordinary app's playback capture
 * records silence. The shell uid may register an audio policy mix directly, and one
 * with only the LOOP_BACK route flag *moves* the matched players to the capturer:
 * the original stops reaching the speaker, exactly like the desktop's stream move.
 * Same hidden-API calls as scrcpy's AudioPlaybackCapture (Apache-2.0).
 *
 * Runs outside the app, as shell (adb for now, Shizuku later):
 *   CLASSPATH=$(pm path org.stemflow.app | cut -d: -f2) \
 *       app_process / org.stemflow.app.Helper PORT UID [UID...]
 * and writes 44.1 kHz stereo s16le PCM to 127.0.0.1:PORT, where the app listens.
 * Exiting (or the app closing the socket) drops the mix and playback returns.
 */
@SuppressLint({"PrivateApi", "DiscouragedPrivateApi", "BlockedPrivateApi", "SoonBlockedPrivateApi"})
public final class Helper {
    static final String SHELL = "com.android.shell";
    public static final int RATE = 44100;

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("usage: Helper PORT UID [UID...]");
            System.exit(2);
        }
        int port = Integer.parseInt(args[0]);
        int[] uids = new int[args.length - 1];
        for (int i = 1; i < args.length; i++) uids[i - 1] = Integer.parseInt(args[i]);

        Looper.prepareMainLooper();
        Context context = fakeContext();
        AudioRecord record = createRecord(context, uids);

        Socket socket = new Socket(InetAddress.getLoopbackAddress(), port);
        socket.setTcpNoDelay(true);
        OutputStream out = socket.getOutputStream();
        record.startRecording();
        System.err.println("stemflow helper: capturing, state=" + record.getRecordingState());
        byte[] buf = new byte[RATE / 20 * 4]; // 50 ms
        while (true) {
            int n = record.read(buf, 0, buf.length);
            if (n <= 0) break;
            try {
                out.write(buf, 0, n);
            } catch (java.io.IOException e) {
                break; // the app went away
            }
        }
        System.exit(0); // drops the mix: the source plays on the speaker again
    }

    private static AudioRecord createRecord(Context context, int[] uids) throws Exception {
        Class<?> ruleClass = Class.forName("android.media.audiopolicy.AudioMixingRule");
        Class<?> ruleBuilderClass = Class.forName("android.media.audiopolicy.AudioMixingRule$Builder");
        Object ruleBuilder = ruleBuilderClass.getConstructor().newInstance();
        ruleBuilderClass.getMethod("setTargetMixRole", int.class)
                .invoke(ruleBuilder, ruleClass.getField("MIX_ROLE_PLAYERS").getInt(null));
        int matchUid = ruleClass.getField("RULE_MATCH_UID").getInt(null);
        Method addMixRule = ruleBuilderClass.getMethod("addMixRule", int.class, Object.class);
        for (int uid : uids) addMixRule.invoke(ruleBuilder, matchUid, uid);
        // Not allowPrivilegedPlaybackCapture(true): AudioMix then refuses any sample
        // rate above 16 kHz. It isn't needed; the shell uid's mix gets Spotify anyway.
        Object rule = ruleBuilderClass.getMethod("build").invoke(ruleBuilder);

        AudioFormat format = new AudioFormat.Builder()
                .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                .setSampleRate(RATE)
                .setChannelMask(AudioFormat.CHANNEL_IN_STEREO)
                .build();
        Class<?> mixClass = Class.forName("android.media.audiopolicy.AudioMix");
        Class<?> mixBuilderClass = Class.forName("android.media.audiopolicy.AudioMix$Builder");
        Object mixBuilder = mixBuilderClass.getConstructor(ruleClass).newInstance(rule);
        mixBuilderClass.getMethod("setFormat", AudioFormat.class).invoke(mixBuilder, format);
        // LOOP_BACK without RENDER: the players are rerouted to us, not duplicated.
        mixBuilderClass.getMethod("setRouteFlags", int.class)
                .invoke(mixBuilder, mixClass.getField("ROUTE_FLAG_LOOP_BACK").getInt(null));
        Object mix = mixBuilderClass.getMethod("build").invoke(mixBuilder);

        Class<?> policyClass = Class.forName("android.media.audiopolicy.AudioPolicy");
        Class<?> policyBuilderClass = Class.forName("android.media.audiopolicy.AudioPolicy$Builder");
        Object policyBuilder = policyBuilderClass.getConstructor(Context.class).newInstance(context);
        policyBuilderClass.getMethod("addMix", mixClass).invoke(policyBuilder, mix);
        Object policy = policyBuilderClass.getMethod("build").invoke(policyBuilder);

        Method register = AudioManager.class.getDeclaredMethod("registerAudioPolicyStatic", policyClass);
        register.setAccessible(true);
        int result = (int) register.invoke(null, policy);
        if (result != 0) throw new RuntimeException("registerAudioPolicy() returned " + result);

        return (AudioRecord) policyClass.getMethod("createAudioRecordSink", mixClass).invoke(policy, mix);
    }

    /** A Context that says it is the shell package, on top of a bare ActivityThread. */
    private static Context fakeContext() throws Exception {
        Class<?> atClass = Class.forName("android.app.ActivityThread");
        Constructor<?> ctor = atClass.getDeclaredConstructor();
        ctor.setAccessible(true);
        Object at = ctor.newInstance();
        setField(atClass, null, "sCurrentActivityThread", at);
        setField(atClass, at, "mSystemThread", true);
        Method getSystemContext = atClass.getDeclaredMethod("getSystemContext");
        getSystemContext.setAccessible(true);
        Context fake = new ShellContext((Context) getSystemContext.invoke(at));

        // Some framework paths ask the ActivityThread who the application is.
        try {
            Class<?> bindClass = Class.forName("android.app.ActivityThread$AppBindData");
            Constructor<?> bindCtor = bindClass.getDeclaredConstructor();
            bindCtor.setAccessible(true);
            Object bind = bindCtor.newInstance();
            ApplicationInfo info = new ApplicationInfo();
            info.packageName = SHELL;
            setField(bindClass, bind, "appInfo", info);
            setField(atClass, at, "mBoundApplication", bind);
            Application app = new Application();
            setField(ContextWrapper.class, app, "mBase", fake);
            setField(atClass, at, "mInitialApplication", app);
        } catch (Throwable t) {
            System.err.println("stemflow helper: app info workaround failed: " + t);
        }
        return fake;
    }

    private static void setField(Class<?> c, Object target, String name, Object value) throws Exception {
        Field f = c.getDeclaredField(name);
        f.setAccessible(true);
        f.set(target, value);
    }

    private static final class ShellContext extends ContextWrapper {
        ShellContext(Context base) { super(base); }
        @Override public String getPackageName() { return SHELL; }
        @Override public String getOpPackageName() { return SHELL; }
        @Override public Context getApplicationContext() { return this; }
        @Override public int getDeviceId() { return 0; }
        @Override public AttributionSource getAttributionSource() {
            return new AttributionSource.Builder(Process.SHELL_UID).setPackageName(SHELL).build();
        }
        @Override public Context createPackageContext(String packageName, int flags) { return this; }
    }
}
