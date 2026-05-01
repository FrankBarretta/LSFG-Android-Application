# Native bridge: JNI methods are looked up by mangled name, must keep class + natives.
-keep class com.lsfg.android.session.NativeBridge { *; }
-keepclasseswithmembernames class * {
    native <methods>;
}

# AIDL-generated stubs for Shizuku user-service IPC.
-keep class com.lsfg.android.shizuku.** { *; }

# Shizuku API uses reflection / dynamic proxies for the manager service binder.
-keep class rikka.shizuku.** { *; }
-keep interface rikka.shizuku.** { *; }
-dontwarn rikka.shizuku.**

# Shizuku spawns ShizukuCaptureUserService in a separate process and instantiates
# it by class name via reflection (Class.forName(...).newInstance()). The class
# name is also passed to bindUserService through ComponentName, which uses the
# obfuscated name. Keep both the class and its no-arg constructor verbatim.
-keep class com.lsfg.android.session.ShizukuCaptureUserService { *; }

# libsu spawns a remote root process and resolves classes by name across the IPC boundary.
-keep class com.topjohnwu.superuser.** { *; }
-keep interface com.topjohnwu.superuser.** { *; }
-dontwarn com.topjohnwu.superuser.**

# RootCaptureService extends libsu's RootService and is instantiated by name in
# the spawned root process. Same constraint as ShizukuCaptureUserService.
-keep class com.lsfg.android.session.RootCaptureService { *; }
-keep class com.lsfg.android.session.RootCaptureService$* { *; }

# Parcelables / Binder stubs declared anywhere in the app.
-keepclassmembers class * implements android.os.Parcelable {
    public static final ** CREATOR;
}

# Strip log calls from the release build to shave a bit more.
-assumenosideeffects class android.util.Log {
    public static *** v(...);
    public static *** d(...);
}
