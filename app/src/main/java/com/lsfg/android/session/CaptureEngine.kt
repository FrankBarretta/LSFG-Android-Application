package com.lsfg.android.session

import android.content.Context
import android.graphics.PixelFormat
import android.hardware.HardwareBuffer
import android.hardware.display.VirtualDisplay
import android.media.ImageReader
import android.media.projection.MediaProjection
import android.os.Handler
import android.os.HandlerThread
import android.os.Process
import android.view.Surface

class CaptureEngine(
    private val ctx: Context,
    private val mediaProjection: MediaProjection,
) {

    private var virtualDisplay: VirtualDisplay? = null

    private var captureThread: HandlerThread? = null
    private var captureHandler: Handler? = null

    private var imageReader: ImageReader? = null

    @Volatile
    private var currentSurface: Surface? = null

    @Volatile
    private var lsfgEnabled = false

    @Volatile
    var frameGenBypass = false

    @Volatile
    private var nativeInputEnabled = true

    companion object {
        private const val TAG = "CaptureEngine"

        // 0.75 = MAIS FPS
        // 1.0 = qualidade máxima
        private const val RESOLUTION_SCALE = 0.75f

        // menor = menos latência
        private const val MAX_IMAGES = 2
    }

    // =========================
    // THREAD
    // =========================

    private fun ensureThread(): Handler {
        val existing = captureHandler
        if (existing != null) return existing

        val t = HandlerThread(
            "lsfg-fast-capture",
            Process.THREAD_PRIORITY_URGENT_DISPLAY
        )

        t.start()

        val h = Handler(t.looper)

        captureThread = t
        captureHandler = h

        return h
    }

    // =========================
    // MIRROR MODE
    // =========================

    @Synchronized
    fun setSurface(
        outputSurface: Surface,
        width: Int,
        height: Int
    ) {

        if (!outputSurface.isValid) return

        nativeInputEnabled = false

        currentSurface = outputSurface

        val scaledW = (width * RESOLUTION_SCALE).toInt()
        val scaledH = (height * RESOLUTION_SCALE).toInt()

        val dpi = ctx.resources.displayMetrics.densityDpi

        val vd = virtualDisplay

        if (vd == null) {

            val handler = ensureThread()

            virtualDisplay = mediaProjection.createVirtualDisplay(
                "lsfg-vd",
                scaledW,
                scaledH,
                dpi,
                0,
                outputSurface,
                null,
                handler
            )

        } else {

            // NÃO REDIMENSIONA
            // Apenas troca surface

            try {
                vd.surface = outputSurface
            } catch (_: Throwable) {
            }
        }
    }

    // =========================
    // LSFG MODE
    // =========================

    @Synchronized
    fun setLsfgMode(
        width: Int,
        height: Int
    ) {

        stopLsfg()

        lsfgEnabled = true
        nativeInputEnabled = true

        val scaledW = (width * RESOLUTION_SCALE).toInt()
        val scaledH = (height * RESOLUTION_SCALE).toInt()

        val handler = ensureThread()

        val reader = ImageReader.newInstance(
            scaledW,
            scaledH,
            PixelFormat.RGBA_8888,
            MAX_IMAGES,
            HardwareBuffer.USAGE_GPU_SAMPLED_IMAGE
        )

        reader.setOnImageAvailableListener({ r ->

            // MAIS FPS:
            // pega somente o frame mais recente

            val img = try {
                r.acquireLatestImage()
            } catch (_: Throwable) {
                null
            } ?: return@setOnImageAvailableListener

            try {

                val hb = img.hardwareBuffer
                    ?: return@setOnImageAvailableListener

                if (nativeInputEnabled) {

                    try {
                        NativeBridge.pushFrame(
                            hb,
                            img.timestamp
                        )
                    } catch (_: Throwable) {
                    }
                }

                hb.close()

            } finally {
                img.close()
            }

        }, handler)

        imageReader = reader

        val dpi = ctx.resources.displayMetrics.densityDpi

        val vd = virtualDisplay

        if (vd == null) {

            virtualDisplay = mediaProjection.createVirtualDisplay(
                "lsfg-fast-vd",
                scaledW,
                scaledH,
                dpi,
                0,
                reader.surface,
                null,
                handler
            )

        } else {

            try {
                vd.surface = reader.surface
            } catch (_: Throwable) {
            }
        }
    }

    // =========================
    // STOP LSFG
    // =========================

    @Synchronized
    private fun stopLsfg() {

        nativeInputEnabled = false

        try {
            virtualDisplay?.surface = null
        } catch (_: Throwable) {
        }

        try {
            imageReader?.close()
        } catch (_: Throwable) {
        }

        imageReader = null
    }

    // =========================
    // CLEAR SURFACE
    // =========================

    @Synchronized
    fun clearSurface() {

        currentSurface = null

        try {
            virtualDisplay?.surface = null
        } catch (_: Throwable) {
        }
    }

    // =========================
    // PAUSE INPUT
    // =========================

    @Synchronized
    fun pauseLsfgInput() {
        nativeInputEnabled = false
    }

    // =========================
    // RESUME INPUT
    // =========================

    @Synchronized
    fun resumeLsfgInput() {
        nativeInputEnabled = true
    }

    // =========================
    // STOP EVERYTHING
    // =========================

    @Synchronized
    fun stop() {

        nativeInputEnabled = false

        try {
            imageReader?.close()
        } catch (_: Throwable) {
        }

        imageReader = null

        try {
            virtualDisplay?.release()
        } catch (_: Throwable) {
        }

        virtualDisplay = null

        try {
            captureThread?.quitSafely()
        } catch (_: Throwable) {
        }

        captureThread = null
        captureHandler = null

        currentSurface = null
    }
}
