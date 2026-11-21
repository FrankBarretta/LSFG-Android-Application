package com.lsfg.android

import android.app.Application
import android.os.Build
import android.util.Log
import org.lsposed.hiddenapibypass.HiddenApiBypass
import com.lsfg.android.session.CrashReporter
import com.lsfg.android.session.LsfgLog

class LsfgApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        // Install the crash reporter as early as possible so even a crash during
        // the first JNI load attempt gets captured (NativeBridge static init
        // already loads the .so; the call below only configures the handler).
        CrashReporter.install(this)
        LsfgLog.init(this)
        // Must run before OverlayManager reflects into ViewRootImpl/SurfaceControl.
        // Measured on Android 13 (Retroid Pocket Flip 2): without it
        // installSkipScreenshot fails and the overlay is captured by MediaProjection.
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            val ok = runCatching { HiddenApiBypass.addHiddenApiExemptions("") }.getOrDefault(false)
            Log.i("lsfg-vk-loop", "HiddenApiBypass exemptions applied=$ok")
        }
    }
}
