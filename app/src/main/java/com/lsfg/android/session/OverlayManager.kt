package com.lsfg.android.session

import android.content.Context
import android.graphics.Color
import android.graphics.PixelFormat
import android.graphics.Region
import android.graphics.SurfaceTexture
import android.os.Build
import android.util.DisplayMetrics
import android.util.Log
import android.view.Gravity
import android.view.Surface
import android.view.TextureView
import android.view.View
import android.view.WindowManager
import android.widget.FrameLayout
import android.widget.TextView
import com.lsfg.android.prefs.LsfgPreferences
import com.lsfg.android.prefs.VsyncRefreshOverride

/**
 * Full-screen overlay that hosts a [TextureView] for the mirrored / LSFG-processed
 * frames.
 *
 * **Why TextureView instead of SurfaceView**: a SurfaceView creates a child BLAST
 * SurfaceControl that lives outside the parent window's surface hierarchy. On strict
 * AOSP builds (e.g. Rockchip Orange Pi 5 Ultra, Android 13) InputDispatcher's
 * BLOCK_UNTRUSTED_TOUCHES filter evaluates that BLAST surface independently, sees
 * an opaque (alpha=1.0) untrusted overlay sitting over the target app, and drops
 * every tap with `Dropping untrusted touch event due to /<uid>` — even when the
 * parent window has an empty touchable region and is itself trusted. The
 * "trusted" bit does NOT propagate to the child SurfaceControl, and the only API
 * to mark it trusted (`SurfaceControl.Transaction.setTrustedOverlay`) is hidden /
 * blocklisted on user builds.
 *
 * TextureView draws into the parent View's hardware layer instead of creating a
 * separate SurfaceControl, so InputDispatcher only sees one window — the parent
 * one, whose touchable region we already publish as empty. The cost is one extra
 * GPU copy per frame relative to SurfaceView; acceptable here because the
 * native render loop already CPU-blits each generated frame to ANativeWindow.
 *
 * Surface lifecycle is event-driven: consumers get [onSurfaceReady] every time a
 * new Surface becomes valid (initial show and every recreation after orientation
 * / immersive-mode changes) and [onSurfaceLost] every time the Surface is torn
 * down. We pin the producer-side buffer to the physical screen size so the
 * Surface dimensions match the VirtualDisplay dimensions exactly — no scaling,
 * no letter-boxing, no off-screen positioning.
 */
class OverlayManager(private val ctx: Context) {

    private var root: FrameLayout? = null
    private var textureView: TextureView? = null
    private var producerSurface: Surface? = null
    private var hostWindowManager: WindowManager? = null
    private var fpsView: TextView? = null
    private var graphView: FrameGraphView? = null
    private var insetsListener: Any? = null
    private var internalInsetsListener: Any? = null

    @Volatile
    private var surfaceReadyListener: ((Surface, Int, Int) -> Unit)? = null

    @Volatile
    private var surfaceLostListener: (() -> Unit)? = null
    private var requestedRefreshRateHz: Float = 0f
    private var overlayWidth: Int = 0
    private var overlayHeight: Int = 0

    /** Callback invoked every time the overlay Surface becomes valid for writing. */
    fun onSurfaceReady(cb: (Surface, Int, Int) -> Unit) {
        surfaceReadyListener = cb
    }

    /** Callback invoked every time the Surface is torn down. */
    fun onSurfaceLost(cb: () -> Unit) {
        surfaceLostListener = cb
    }

    fun show() {
        if (root != null) return

        // Two host modes, controlled by the user's `trustedOverlay` preference:
        //
        //  - TYPE_APPLICATION_OVERLAY (default): the overlay sits below the
        //    system bars (status bar, navigation bar, notification shade) so
        //    the user can still pull down notifications and tap nav buttons.
        //    Works on most devices because BLOCK_UNTRUSTED_TOUCHES is
        //    permissive on Pixel/Samsung/Xiaomi/etc.
        //
        //  - TYPE_ACCESSIBILITY_OVERLAY (opt-in via preference, requires the
        //    LsfgAccessibilityService to be bound): the overlay becomes a
        //    trusted overlay so InputDispatcher's BLOCK_UNTRUSTED_TOUCHES
        //    filter does not drop tap pass-through on strict AOSP builds (e.g.
        //    Rockchip Orange Pi 5 Ultra, Android 13). Trade-off: a11y overlays
        //    are forced into a system layer family ABOVE the status / nav
        //    bars, so the user loses access to those bars while the session is
        //    running. Combined with TextureView (which avoids the child BLAST
        //    SurfaceControl that would re-introduce an untrusted occluder),
        //    this is the only path that makes pass-through actually work on
        //    those devices.
        val prefs = LsfgPreferences(ctx).load()
        val a11y = LsfgAccessibilityService.instance
        val useTrusted = prefs.trustedOverlay && a11y != null
        val hostCtx: Context = if (useTrusted) a11y!! else ctx
        val wm = hostCtx.getSystemService(Context.WINDOW_SERVICE) as WindowManager
        hostWindowManager = wm

        val metrics = DisplayMetrics()
        @Suppress("DEPRECATION")
        wm.defaultDisplay.getRealMetrics(metrics)
        @Suppress("DEPRECATION")
        requestedRefreshRateHz = wm.defaultDisplay.supportedModes
            .maxOfOrNull { it.refreshRate }
            ?: wm.defaultDisplay.refreshRate
        // Let the native pacer align its sleeps to this display's vsync.
        // Without it two consecutive unlockAndPost calls (gen + real, or two
        // generated) can land in the same SurfaceFlinger slot and one gets
        // held an extra vsync — visible as periodic stutter. The user can
        // disable this or force a specific refresh rate via settings.
        val effectiveHz = if (!prefs.vsyncAlignmentEnabled) {
            0f
        } else if (prefs.vsyncRefreshOverride != VsyncRefreshOverride.AUTO) {
            prefs.vsyncRefreshOverride.hz.toFloat()
        } else {
            requestedRefreshRateHz
        }
        if (effectiveHz > 0f) {
            val periodNs = (1_000_000_000.0 / effectiveHz).toLong()
            runCatching { NativeBridge.setVsyncPeriodNs(periodNs) }
        } else {
            runCatching { NativeBridge.setVsyncPeriodNs(0L) }
        }
        val screenW = metrics.widthPixels
        val screenH = metrics.heightPixels
        overlayWidth = screenW
        overlayHeight = screenH
        Log.i(TAG, "Showing overlay at ${screenW}x${screenH} targetRefresh=${requestedRefreshRateHz}Hz")

        val layoutType = when {
            useTrusted -> WindowManager.LayoutParams.TYPE_ACCESSIBILITY_OVERLAY
            Build.VERSION.SDK_INT >= Build.VERSION_CODES.O ->
                WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY
            else -> @Suppress("DEPRECATION")
            WindowManager.LayoutParams.TYPE_SYSTEM_ALERT
        }
        Log.i(TAG, "Overlay host=${if (useTrusted) "a11y/TRUSTED" else "app/UNTRUSTED"}")

        // No FLAG_NOT_TOUCHABLE: that flag, combined with TYPE_APPLICATION_OVERLAY, is
        // what triggers the Android 12+ 0.8-alpha clamp. Pass-through is handled by an
        // empty touchable region (installed right after addView).
        val flags = WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL or
                WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN or
                WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS or
                WindowManager.LayoutParams.FLAG_HARDWARE_ACCELERATED or
                WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON

        val params = WindowManager.LayoutParams(
            screenW,
            screenH,
            layoutType,
            flags,
            PixelFormat.OPAQUE,
        ).apply {
            gravity = Gravity.TOP or Gravity.START
            x = 0
            y = 0
            alpha = 1.0f
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                layoutInDisplayCutoutMode =
                    WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_ALWAYS
            }
            @Suppress("DEPRECATION")
            systemUiVisibility = (View.SYSTEM_UI_FLAG_LAYOUT_STABLE
                    or View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                    or View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                    or View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                    or View.SYSTEM_UI_FLAG_FULLSCREEN
                    or View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY)
        }

        // FrameLayout background stays transparent — TextureView composites into
        // its parent's hardware layer, but we still want no opaque fill behind it
        // before the first frame arrives so the loading status text remains
        // visible against the underlying app instead of a black slab.
        val layout = FrameLayout(ctx)
        val tex = TextureView(ctx)
        tex.isOpaque = true
        tex.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
            override fun onSurfaceTextureAvailable(st: SurfaceTexture, width: Int, height: Int) {
                // Pin the producer-side buffer size to the physical screen so the
                // VirtualDisplay's frames don't get scaled by the consumer.
                st.setDefaultBufferSize(screenW, screenH)
                val s = Surface(st)
                producerSurface = s
                syncOverlayGeometry()
                requestMaxRefreshRate(s)
                Log.i(TAG, "TextureView surface available ${width}x${height} valid=${s.isValid}")
                if (s.isValid) {
                    surfaceReadyListener?.invoke(s, overlayWidth, overlayHeight)
                }
            }

            override fun onSurfaceTextureSizeChanged(st: SurfaceTexture, width: Int, height: Int) {
                st.setDefaultBufferSize(screenW, screenH)
                val s = producerSurface
                syncOverlayGeometry()
                if (s != null) {
                    requestMaxRefreshRate(s)
                    Log.i(TAG, "TextureView size changed ${width}x${height}")
                    if (s.isValid) {
                        surfaceReadyListener?.invoke(s, overlayWidth, overlayHeight)
                    }
                }
            }

            override fun onSurfaceTextureDestroyed(st: SurfaceTexture): Boolean {
                Log.i(TAG, "TextureView surface destroyed")
                surfaceLostListener?.invoke()
                runCatching { producerSurface?.release() }
                producerSurface = null
                // Returning true tells TextureView it can release the SurfaceTexture
                // immediately; we have no off-thread producer holding a reference
                // beyond the native render loop, which has already been notified
                // via surfaceLostListener and detaches before this returns.
                return true
            }

            override fun onSurfaceTextureUpdated(st: SurfaceTexture) = Unit
        }
        val fps = TextView(ctx).apply {
            text = ""
            setTextColor(Color.WHITE)
            setBackgroundColor(0x80000000.toInt())
            textSize = 14f
            setPadding(16, 8, 16, 8)
            visibility = View.GONE
        }
        layout.addView(
            tex,
            FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT,
            ),
        )
        layout.addView(
            fps,
            FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.WRAP_CONTENT,
                FrameLayout.LayoutParams.WRAP_CONTENT,
                Gravity.TOP or Gravity.START,
            ).apply {
                topMargin = 24
                leftMargin = 24
            },
        )

        val graph = FrameGraphView(ctx).apply {
            visibility = View.GONE
        }
        val graphWidthPx = (metrics.density * 220f).toInt()
        val graphHeightPx = (metrics.density * 80f).toInt()
        layout.addView(
            graph,
            FrameLayout.LayoutParams(
                graphWidthPx,
                graphHeightPx,
                Gravity.TOP or Gravity.START,
            ).apply {
                // Position just below the FPS text so both stay in the top-left cluster.
                topMargin = 112
                leftMargin = 24
            },
        )

        wm.addView(layout, params)
        // Two complementary pass-through mechanisms — we install both because each
        // is needed on a different subset of devices:
        //   1) AttachedSurfaceControl.setTouchableRegion(empty) — the modern API
        //      (public on API 33+, reflective on 29–32). Most reliable on Pixel /
        //      AOSP-like ROMs.
        //   2) ViewTreeObserver.OnComputeInternalInsetsListener with
        //      TOUCHABLE_INSETS_REGION — the legacy SystemUI pattern that some
        //      OEMs (Samsung One UI, Xiaomi HyperOS, OPPO ColorOS) honour even
        //      when (1) is silently rejected.
        // We do NOT use FLAG_NOT_TOUCHABLE, which would re-enable the Android 12+
        // 0.8-alpha clamp.
        insetsListener = installEmptyTouchableRegion(layout)
        internalInsetsListener = installEmptyInternalInsets(layout)
        root = layout
        fpsView = fps
        graphView = graph
        textureView = tex
    }

    /**
     * Re-asserts the overlay as the topmost window. Call this after the target app is
     * launched so the new foreground activity doesn't leave a stale z-order with our
     * overlay stuck behind it.
     */
    fun bringToFront() {
        val r = root ?: return
        val wm = hostWindowManager ?: return
        if (!r.isAttachedToWindow) return
        val lp = r.layoutParams as? WindowManager.LayoutParams ?: return
        runCatching { wm.updateViewLayout(r, lp) }
            .onFailure { Log.w(TAG, "bringToFront updateViewLayout failed", it) }
    }

    /**
     * Re-reads display metrics and resizes the overlay window to match a new
     * orientation / display configuration. Idempotent; safe to call multiple
     * times for a single rotation. The actual `wm.updateViewLayout` is posted
     * to the root view's main-thread looper to avoid racing with WindowManager
     * while it is still distributing the configuration change.
     */
    fun onDisplayConfigurationChanged() {
        val r = root ?: return
        r.post { syncOverlayGeometry() }
    }

    fun updateStatus(line: String) {
        // Status text was removed from the overlay; keep the API as a no-op so
        // existing call sites in the foreground service continue to compile.
    }

    fun setFpsVisible(visible: Boolean) {
        fpsView?.post { fpsView?.visibility = if (visible) View.VISIBLE else View.GONE }
    }

    fun updateFps(capturedFps: Float, postedFps: Float) {
        val v = fpsView ?: return
        val text = "real ${"%.1f".format(capturedFps)} fps · total ${"%.1f".format(postedFps)} fps"
        v.post { v.text = text }
    }

    fun setFrameGraphVisible(visible: Boolean) {
        val g = graphView ?: return
        g.post {
            g.visibility = if (visible) View.VISIBLE else View.GONE
            if (!visible) g.reset()
        }
    }

    fun pushFrameGraphSample(realFps: Float, generatedFps: Float) {
        val g = graphView ?: return
        g.post { g.pushSample(realFps, generatedFps) }
    }

    fun hide() {
        val r = root ?: return
        insetsListener?.let { removeEmptyTouchableRegion(r, it) }
        insetsListener = null
        internalInsetsListener?.let { removeEmptyInternalInsets(r, it) }
        internalInsetsListener = null
        val wm = hostWindowManager
        if (wm != null) {
            runCatching { wm.removeView(r) }
        }
        runCatching { NativeBridge.setVsyncPeriodNs(0L) }
        runCatching { producerSurface?.release() }
        producerSurface = null
        root = null
        textureView = null
        fpsView = null
        hostWindowManager = null
    }

    /**
     * Publishes an empty touchable region on the overlay's root surface so
     * InputDispatcher skips the window and events fall through to the game below.
     *
     * Primary path (API 33+): public `View.getRootSurfaceControl().setTouchableRegion()`.
     * Legacy path (API 29-32): `getRootSurfaceControl()` is `@hide` but callable via
     *   reflection (not in the blocklist). `AttachedSurfaceControl.setTouchableRegion`
     *   has been present since API 29.
     * The call must happen after the view is attached to a window, so we hook it
     * into `onAttachedToWindow` / `OnAttachStateChangeListener`.
     *
     * Returns the attach listener (so we can detach it in hide()) or null if the
     * reflective lookup failed on a very old device.
     */
    private fun installEmptyTouchableRegion(host: View): Any? {
        val applyEmptyRegion: () -> Unit = {
            runCatching {
                val rootSc = when {
                    Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU -> host.rootSurfaceControl
                    else -> host.javaClass
                        .getMethod("getRootSurfaceControl")
                        .invoke(host)
                }
                if (rootSc == null) {
                    Log.w(TAG, "rootSurfaceControl is null; window may not be attached yet")
                } else {
                    val setTouchableRegion = rootSc.javaClass
                        .getMethod("setTouchableRegion", Region::class.java)
                    setTouchableRegion.invoke(rootSc, Region())
                    Log.i(TAG, "Empty touchable region applied (api=${Build.VERSION.SDK_INT})")
                }
            }.onFailure { Log.w(TAG, "setTouchableRegion(empty) failed", it) }
        }

        if (host.isAttachedToWindow) {
            applyEmptyRegion()
            return Unit
        }
        val attachListener = object : View.OnAttachStateChangeListener {
            override fun onViewAttachedToWindow(v: View) {
                applyEmptyRegion()
            }
            override fun onViewDetachedFromWindow(v: View) = Unit
        }
        host.addOnAttachStateChangeListener(attachListener)
        return attachListener
    }

    private fun removeEmptyTouchableRegion(host: View, listener: Any) {
        if (listener is View.OnAttachStateChangeListener) {
            runCatching { host.removeOnAttachStateChangeListener(listener) }
                .onFailure { Log.w(TAG, "removeOnAttachStateChangeListener failed", it) }
        }
    }

    /**
     * Belt-and-braces touch pass-through using the SystemUI-canonical
     * `ViewTreeObserver.OnComputeInternalInsetsListener` + `TOUCHABLE_INSETS_REGION`.
     * Both the listener interface and `InternalInsetsInfo` are `@hide` in the public
     * SDK (stable since API 1), so the implementation is built reflectively. On
     * devices where this succeeds, InputDispatcher excludes our window from the
     * touchable region and events fall through to the app underneath even if
     * setTouchableRegion(empty) was silently rejected.
     *
     * Returns the proxy listener (so it can be detached in [hide]) or null on failure.
     */
    private fun installEmptyInternalInsets(host: View): Any? {
        return runCatching {
            val internalInsetsInfoCls = Class.forName("android.view.ViewTreeObserver\$InternalInsetsInfo")
            val listenerCls = Class.forName("android.view.ViewTreeObserver\$OnComputeInternalInsetsListener")
            val touchableInsetsRegion = internalInsetsInfoCls
                .getField("TOUCHABLE_INSETS_REGION")
                .getInt(null)
            val setTouchableInsets = internalInsetsInfoCls
                .getMethod("setTouchableInsets", Int::class.javaPrimitiveType)
            val touchableRegionField = internalInsetsInfoCls.getField("touchableRegion")

            val proxy = java.lang.reflect.Proxy.newProxyInstance(
                listenerCls.classLoader,
                arrayOf(listenerCls),
            ) { _, method, args ->
                if (method.name == "onComputeInternalInsets" && args != null && args.isNotEmpty()) {
                    val info = args[0]
                    runCatching {
                        setTouchableInsets.invoke(info, touchableInsetsRegion)
                        (touchableRegionField.get(info) as? Region)?.setEmpty()
                    }
                }
                null
            }
            val addMethod = host.viewTreeObserver.javaClass
                .getMethod("addOnComputeInternalInsetsListener", listenerCls)
            addMethod.invoke(host.viewTreeObserver, proxy)
            Log.i(TAG, "OnComputeInternalInsetsListener pass-through installed")
            proxy
        }.onFailure { Log.w(TAG, "installEmptyInternalInsets failed", it) }.getOrNull()
    }

    private fun removeEmptyInternalInsets(host: View, listener: Any) {
        runCatching {
            val listenerCls = Class.forName("android.view.ViewTreeObserver\$OnComputeInternalInsetsListener")
            val removeMethod = host.viewTreeObserver.javaClass
                .getMethod("removeOnComputeInternalInsetsListener", listenerCls)
            removeMethod.invoke(host.viewTreeObserver, listener)
        }.onFailure { Log.w(TAG, "removeOnComputeInternalInsetsListener failed", it) }
    }

    private fun requestMaxRefreshRate(surface: Surface) {
        if (!surface.isValid || requestedRefreshRateHz <= 0f || Build.VERSION.SDK_INT < Build.VERSION_CODES.R) {
            return
        }
        runCatching {
            surface.setFrameRate(
                requestedRefreshRateHz,
                Surface.FRAME_RATE_COMPATIBILITY_FIXED_SOURCE,
                Surface.CHANGE_FRAME_RATE_ALWAYS,
            )
        }.onFailure { Log.w(TAG, "setFrameRate(${requestedRefreshRateHz}Hz) failed", it) }
    }

    private fun syncOverlayGeometry() {
        val wm = hostWindowManager ?: return
        val r = root ?: return
        if (!r.isAttachedToWindow) {
            // The window has already been torn down (or hasn't finished attaching).
            // updateViewLayout in either state throws on some OEMs.
            return
        }
        val metrics = DisplayMetrics()
        @Suppress("DEPRECATION")
        wm.defaultDisplay.getRealMetrics(metrics)
        val newW = metrics.widthPixels
        val newH = metrics.heightPixels
        if (newW == overlayWidth && newH == overlayHeight) return

        overlayWidth = newW
        overlayHeight = newH

        val lp = r.layoutParams as? WindowManager.LayoutParams
        if (lp != null) {
            lp.width = newW
            lp.height = newH
            runCatching { wm.updateViewLayout(r, lp) }
                .onFailure { Log.w(TAG, "updateViewLayout(${newW}x${newH}) failed", it) }
        }
        runCatching { textureView?.surfaceTexture?.setDefaultBufferSize(newW, newH) }
            .onFailure { Log.w(TAG, "setDefaultBufferSize(${newW}x${newH}) failed", it) }
        Log.i(TAG, "Overlay geometry synced to ${newW}x${newH}")
    }

    companion object {
        private const val TAG = "OverlayManager"
    }
}
