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

# libsu spawns a remote root process and resolves classes by name across the IPC boundary.
-keep class com.topjohnwu.superuser.** { *; }
-keep interface com.topjohnwu.superuser.** { *; }
-dontwarn com.topjohnwu.superuser.**

# Parcelables / Binder stubs declared anywhere in the app.
-keepclassmembers class * implements android.os.Parcelable {
    public static final ** CREATOR;
}

# Strip log calls from the release build to shave a bit more.
-assumenosideeffects class android.util.Log {
    public static *** v(...);
    public static *** d(...);
}
