package com.lsfg.android.benchmark

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.util.Log
import com.lsfg.android.prefs.LsfgPreferences
import com.lsfg.android.prefs.NpuPostProcessingPreset
import com.lsfg.android.prefs.CpuPostProcessingPreset
import com.lsfg.android.prefs.VsyncRefreshOverride
import java.io.File
import java.util.concurrent.atomic.AtomicReference
import kotlin.concurrent.thread

/**
 * Orchestrates a 3-run benchmark on top of an already-active LSFG session.
 *
 * Lifecycle:
 *   1. The LsfgForegroundService starts a normal session (MediaProjection consent,
 *      target app launched, framegen active).
 *   2. The service sees EXTRA_BENCHMARK_REQUESTED=true and calls [start] passing
 *      a [Hooks] object that lets us trigger context reinit and read render-size.
 *   3. We snapshot the user's prefs, overwrite them with the benchmark preset,
 *      then loop over MULTIPLIERS — for each: set multiplier in prefs, request
 *      reinit (the service path used by the live drawer), warmup, sample, store.
 *   4. After the last run we restore prefs and persist the report file. The
 *      service is then asked to stop the session and surface the share intent.
 *
 * State is held as `AtomicReference<RunState>` so the UI (which lives in the
 * Activity / a separate Compose screen) can poll progress without locking.
 */
object BenchmarkController {

    private const val TAG = "BenchmarkController"

    sealed class State {
        data object Idle : State()
        data class Running(
            val runIndex: Int,            // 0-based: 0 = first run (×2)
            val totalRuns: Int,
            val multiplier: Int,
            val phase: Phase,
            val phaseStartedAtMs: Long,
            val phaseDurationMs: Long,
        ) : State()
        data class Completed(
            val reportFile: File,
            val results: List<BenchmarkRunResult>,
            val targetPackage: String?,
            val renderWidth: Int,
            val renderHeight: Int,
        ) : State()
        data class Failed(val message: String) : State()
    }

    enum class Phase { WARMUP, SAMPLING }

    /** Hooks the service exposes so the controller can drive lifecycle events
     *  it doesn't own (context reinit, render-size lookup, session stop). */
    interface Hooks {
        /** Trigger the same code path the live drawer uses: re-read prefs and
         *  call NativeBridge.destroyContext + initContext on a worker thread.
         *  Returns immediately; reinit completes asynchronously. */
        fun requestReinit()

        /** Currently active render width (set by the service after initContext).
         *  Returns 0 if not yet known. */
        fun activeRenderWidth(): Int
        fun activeRenderHeight(): Int

        /** The current foreground target package, if any. */
        fun targetPackage(): String?

        /** Display vsync period in ns (matches NativeBridge.setVsyncPeriodNs).
         *  Used by BenchmarkRunner for vsync-alignment metrics. */
        fun vsyncPeriodNs(): Long

        /** Called once after the final run to surface the report file to the
         *  user. The service is expected to stop the session and post a
         *  notification with the share intent built from this file. */
        fun onCompleted(state: State.Completed)

        /** Called on early termination (cancel / unrecoverable error). */
        fun onFailed(state: State.Failed)
    }

    private val state = AtomicReference<State>(State.Idle)

    /** Live-reading state for UI bindings. */
    fun currentState(): State = state.get()

    /** True while a benchmark is actively collecting samples. Used by the
     *  service to gate behaviour (e.g. don't allow the drawer to mutate prefs
     *  mid-run). */
    fun isRunning(): Boolean = state.get() is State.Running

    /**
     * Start the 3-run benchmark. Must be called from a thread other than the
     * Main looper, OR pass [forceWorkerThread]=true to launch a worker. The
     * actual benchmark loop sleeps between samples, so it can't run on Main.
     */
    fun start(ctx: Context, hooks: Hooks) {
        val current = state.get()
        if (current is State.Running) {
            Log.w(TAG, "start() called while already running ($current); ignoring")
            return
        }
        thread(name = "lsfg-benchmark", isDaemon = true) {
            try {
                runWorker(ctx.applicationContext, hooks)
            } catch (t: Throwable) {
                Log.e(TAG, "benchmark worker crashed", t)
                val failure = State.Failed("Worker error: ${t.javaClass.simpleName}: ${t.message}")
                state.set(failure)
                postToMain { hooks.onFailed(failure) }
            }
        }
    }

    /** Cancel an in-flight benchmark. Causes the worker to exit at the next
     *  sample tick and report a failure. Restoring prefs is the worker's job. */
    fun cancel() {
        cancelRequested = true
    }

    @Volatile private var cancelRequested: Boolean = false

    private fun runWorker(ctx: Context, hooks: Hooks) {
        cancelRequested = false
        val prefs = LsfgPreferences(ctx)
        val originalConfig = prefs.load()

        // Snapshot now so even an early failure restores the user's prefs.
        val savedMultiplier = originalConfig.multiplier
        val savedFlowScale = originalConfig.flowScale
        val savedPerformance = originalConfig.performanceMode
        val savedAntiArtifacts = originalConfig.antiArtifacts
        val savedNpuEnabled = originalConfig.npuPostProcessingEnabled
        val savedNpuPreset = originalConfig.npuPostProcessingPreset
        val savedCpuEnabled = originalConfig.cpuPostProcessingEnabled
        val savedCpuPreset = originalConfig.cpuPostProcessingPreset
        val savedGpuEnabled = originalConfig.gpuPostProcessingEnabled
        val savedVsyncOverride = originalConfig.vsyncRefreshOverride

        try {
            // Apply the fixed benchmark preset (everything except multiplier).
            prefs.setFlowScale(BenchmarkConfig.FLOW_SCALE)
            prefs.setPerformance(BenchmarkConfig.PERFORMANCE_MODE)
            prefs.setAntiArtifacts(false)
            prefs.setNpuPostProcessingEnabled(false)
            prefs.setNpuPostProcessingPreset(NpuPostProcessingPreset.OFF)
            prefs.setCpuPostProcessingEnabled(false)
            prefs.setCpuPostProcessingPreset(CpuPostProcessingPreset.OFF)
            prefs.setGpuPostProcessingEnabled(false)
            // AUTO honours the display's max refresh — refresh rate is selected
            // at the system level (via the WindowManager.LayoutParams set by
            // the launching Activity), so we don't change it from here. The
            // MainActivity already requests a high refresh hint when the
            // benchmark route is on screen (see BenchmarkScreen).
            prefs.setVsyncRefreshOverride(VsyncRefreshOverride.AUTO)

            val results = ArrayList<BenchmarkRunResult>(BenchmarkConfig.MULTIPLIERS.size)
            val benchmarkStartMs = System.currentTimeMillis()

            for ((idx, mult) in BenchmarkConfig.MULTIPLIERS.withIndex()) {
                if (cancelRequested) {
                    val cancelState = State.Failed("Cancelled by user during run ${idx + 1}")
                    state.set(cancelState)
                    postToMain { hooks.onFailed(cancelState) }
                    return
                }
                Log.i(TAG, "starting run ${idx + 1}/${BenchmarkConfig.MULTIPLIERS.size} multiplier=$mult")
                prefs.setMultiplier(mult)
                hooks.requestReinit()

                publishRunning(
                    runIndex = idx,
                    totalRuns = BenchmarkConfig.MULTIPLIERS.size,
                    multiplier = mult,
                    phase = Phase.WARMUP,
                    phaseDurationMs = BenchmarkConfig.WARMUP_MS,
                )
                sleepInterruptibly(BenchmarkConfig.WARMUP_MS)
                if (cancelRequested) break

                publishRunning(
                    runIndex = idx,
                    totalRuns = BenchmarkConfig.MULTIPLIERS.size,
                    multiplier = mult,
                    phase = Phase.SAMPLING,
                    phaseDurationMs = BenchmarkConfig.RUN_DURATION_SEC * 1000L,
                )
                val result = BenchmarkRunner.collect(
                    multiplier = mult,
                    flowScale = BenchmarkConfig.FLOW_SCALE,
                    performanceMode = BenchmarkConfig.PERFORMANCE_MODE,
                    durationMs = BenchmarkConfig.RUN_DURATION_SEC * 1000L,
                    sampleIntervalMs = BenchmarkConfig.SAMPLE_INTERVAL_MS,
                    nominalVsyncPeriodNs = hooks.vsyncPeriodNs(),
                    shouldStop = { cancelRequested },
                )
                results.add(result)
                Log.i(TAG, "run ${idx + 1} complete: posted=${result.totalPostedFrames} " +
                    "fps=${"%.2f".format(result.postedFps)} stalls=${result.stallCount}")
            }

            if (cancelRequested) {
                val cancelState = State.Failed("Cancelled by user")
                state.set(cancelState)
                postToMain { hooks.onFailed(cancelState) }
                return
            }

            val benchmarkEndMs = System.currentTimeMillis()
            val text = BenchmarkLogWriter.format(
                ctx = ctx,
                results = results,
                startedAtMs = benchmarkStartMs,
                endedAtMs = benchmarkEndMs,
                targetPackage = hooks.targetPackage(),
                renderWidth = hooks.activeRenderWidth(),
                renderHeight = hooks.activeRenderHeight(),
            )
            val file = BenchmarkLogWriter.write(ctx, text)
            val completed = State.Completed(
                reportFile = file,
                results = results,
                targetPackage = hooks.targetPackage(),
                renderWidth = hooks.activeRenderWidth(),
                renderHeight = hooks.activeRenderHeight(),
            )
            state.set(completed)
            postToMain { hooks.onCompleted(completed) }
        } finally {
            // Restore user prefs unconditionally. Reinit is NOT triggered here
            // because the service is going to stop the session right after the
            // share intent fires; a reinit now would just wastefully rebuild
            // the native context for a few ms before tear-down.
            runCatching {
                prefs.setMultiplier(savedMultiplier)
                prefs.setFlowScale(savedFlowScale)
                prefs.setPerformance(savedPerformance)
                prefs.setAntiArtifacts(savedAntiArtifacts)
                prefs.setNpuPostProcessingEnabled(savedNpuEnabled)
                prefs.setNpuPostProcessingPreset(savedNpuPreset)
                prefs.setCpuPostProcessingEnabled(savedCpuEnabled)
                prefs.setCpuPostProcessingPreset(savedCpuPreset)
                prefs.setGpuPostProcessingEnabled(savedGpuEnabled)
                prefs.setVsyncRefreshOverride(savedVsyncOverride)
            }.onFailure { Log.w(TAG, "restoring user prefs failed", it) }
        }
    }

    /** Reset to Idle. Caller (UI) invokes this after consuming a Completed/Failed
     *  state so a future run can start fresh. */
    fun acknowledge() {
        state.set(State.Idle)
    }

    private fun publishRunning(
        runIndex: Int,
        totalRuns: Int,
        multiplier: Int,
        phase: Phase,
        phaseDurationMs: Long,
    ) {
        state.set(
            State.Running(
                runIndex = runIndex,
                totalRuns = totalRuns,
                multiplier = multiplier,
                phase = phase,
                phaseStartedAtMs = System.currentTimeMillis(),
                phaseDurationMs = phaseDurationMs,
            )
        )
    }

    private fun sleepInterruptibly(durationMs: Long) {
        val deadline = System.currentTimeMillis() + durationMs
        while (true) {
            val now = System.currentTimeMillis()
            if (now >= deadline || cancelRequested) return
            val remaining = (deadline - now).coerceAtMost(50L)
            try {
                Thread.sleep(remaining)
            } catch (_: InterruptedException) {
                Thread.currentThread().interrupt()
                return
            }
        }
    }

    private val mainHandler = Handler(Looper.getMainLooper())
    private fun postToMain(block: () -> Unit) = mainHandler.post(block)
}
