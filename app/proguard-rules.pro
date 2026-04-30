# Native bridge: JNI methods are looked up by mangled name, must keep class + natives.
-keep class com.lsfg.android.session.NativeBridge { *; }
-keepclasseswithmembernames class * {
    native <methods>;
}

# Service / Activity / Receiver / AccessibilityService / Application: loaded by
# name from the manifest by the system.
-keep class * extends android.app.Service
-keep class * extends android.app.Activity
-keep class * extends android.app.Application
-keep class * extends android.content.BroadcastReceiver
-keep class * extends android.content.ContentProvider
-keep class * extends android.accessibilityservice.AccessibilityService

# Shizuku uses AIDL + reflection on the provider/binder.
-keep class rikka.shizuku.** { *; }
-keep interface rikka.shizuku.** { *; }
-keep class moe.shizuku.** { *; }
-dontwarn rikka.shizuku.**

# AIDL stubs for our Shizuku/Root bridge.
-keep class com.lsfg.android.shizuku.** { *; }
-keep interface com.lsfg.android.shizuku.** { *; }

# libsu (root) loads RootService implementations by name across processes.
-keep class com.topjohnwu.superuser.** { *; }
-keep interface com.topjohnwu.superuser.** { *; }
-keep class com.lsfg.android.session.RootCaptureService { *; }
-keep class com.lsfg.android.session.ShizukuCaptureUserService { *; }
-dontwarn com.topjohnwu.superuser.**

# Compose runtime needs Signature/InnerClasses for state-handling reflection.
-keepattributes *Annotation*, Signature, InnerClasses, EnclosingMethod, SourceFile, LineNumberTable

# Parcelable CREATOR fields are read by the system by name.
-keepclassmembers class * implements android.os.Parcelable {
    public static final ** CREATOR;
}

# Stop R8 from stripping the line numbers we use to map native crash reports.
-renamesourcefileattribute SourceFile

# Strip log calls from the release build to shave a bit more.
-assumenosideeffects class android.util.Log {
    public static *** v(...);
    public static *** d(...);
}
