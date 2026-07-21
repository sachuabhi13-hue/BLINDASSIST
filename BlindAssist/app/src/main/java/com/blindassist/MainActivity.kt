package com.blindassist

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import android.view.View
import android.view.WindowManager
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.nio.FloatBuffer
import java.util.Locale
import java.util.UUID
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

// ═══════════════════════════════════════════════════════════════════════════
// PATH ANALYSIS RESULT
// Contains both YOLO detections + wall/surface analysis
// ═══════════════════════════════════════════════════════════════════════════

enum class PathStatus {
    CLEAR,           // nothing ahead, safe to walk
    WALL_VERY_CLOSE, // surface fills >70% of center frame = wall right in front
    WALL_CLOSE,      // surface fills 40-70% = wall getting close
    OBSTACLE_DANGER, // known YOLO object < 1m
    OBSTACLE_WARNING // known YOLO object < 2.5m
}

data class FrameAnalysis(
    val detections    : List<DetectedObject>,
    val pathStatus    : PathStatus,
    val blockCoverage : Float,  // 0.0-1.0 how much center is covered by any surface
    val leftOpen      : Boolean,
    val rightOpen     : Boolean,
    val centerOpen    : Boolean
)

// ═══════════════════════════════════════════════════════════════════════════
// DETECTED OBJECT
// ═══════════════════════════════════════════════════════════════════════════

data class DetectedObject(
    val label      : String,
    val confidence : Float,
    val box        : RectF,
    val distanceM  : Float,
    val zone       : String
) {
    fun speak(): String {
        val d = if (distanceM > 0f) "${fmt(distanceM)} metres" else "nearby"
        return "$label $d on your $zone"
    }
    private fun fmt(v: Float) = String.format("%.1f", v)
}

// ═══════════════════════════════════════════════════════════════════════════
// YOLOV8 DETECTOR
// ═══════════════════════════════════════════════════════════════════════════

class Detector(context: Context) {

    companion object {
        private const val TAG   = "Detector"
        private const val MODEL = "yolov8n.onnx"
        private const val SZ    = 640
        private const val CONF  = 0.30f  // LOWERED from 0.40 to catch doors/tables better
        private const val IOU   = 0.45f
        private const val FOCAL = 580f

        // Full COCO 80 classes + extra useful ones
        val LABELS = listOf(
            "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
            "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
            "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
            "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
            "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
            "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
            "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
            "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
            "remote","keyboard","cell phone","microwave","oven","toaster","sink",
            "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
            "toothbrush"
        )

        // Real-world heights in metres for distance estimation
        private val HEIGHTS = mapOf(
            "person"       to 1.70f, "car"          to 1.50f, "bus"      to 3.50f,
            "truck"        to 3.00f, "chair"        to 0.90f, "bottle"   to 0.25f,
            "cup"          to 0.12f, "laptop"       to 0.30f, "bicycle"  to 1.00f,
            "motorcycle"   to 1.20f, "cat"          to 0.35f, "dog"      to 0.55f,
            "couch"        to 0.90f, "dining table" to 0.75f, "bed"      to 0.55f,
            "traffic light" to 3.00f, "stop sign"    to 2.00f, "toilet"   to 0.70f,
            "refrigerator" to 1.80f, "tv"           to 0.60f, "sink"     to 0.80f,
            "bench"        to 0.50f, "backpack"     to 0.50f, "suitcase" to 0.65f,
            "clock"        to 0.30f, "vase"         to 0.35f, "book"     to 0.25f
        )

        // All objects that block movement
        val OBSTACLES = setOf(
            "person","bicycle","car","motorcycle","bus","truck","chair","couch",
            "bed","dining table","dog","cat","suitcase","backpack","potted plant",
            "toilet","refrigerator","tv","bench","sink","oven","microwave"
        )

        // Objects that indicate a possible exit/path
        val EXIT_HINTS = setOf(
            "door","exit sign","stairs","window","gate","archway"
        )

        // Large flat objects that are actually furniture/surfaces
        // used to infer walls/surfaces nearby
        val SURFACE_OBJECTS = setOf(
            "dining table","bed","couch","chair","desk","bench","floor","wall",
            "refrigerator","bookshelf","cabinet","wardrobe","tv","door"
        )
    }

    private val env    : OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        val bytes = context.assets.open(MODEL).readBytes()
        session   = env.createSession(bytes, OrtSession.SessionOptions())
        Log.d(TAG, "Model loaded OK — ${session.inputNames}")
    }

    fun run(bitmap: Bitmap): List<DetectedObject> {
        val scaled  = Bitmap.createScaledBitmap(bitmap, SZ, SZ, true)
        val tensor  = toTensor(scaled)
        val results = session.run(mapOf(session.inputNames.first() to tensor))
        val dets    = parseOutput(results[0].value, bitmap.height, bitmap.width)
        tensor.close(); results.close()
        return dets
    }

    private fun toTensor(bmp: Bitmap): OnnxTensor {
        val px  = IntArray(SZ * SZ)
        bmp.getPixels(px, 0, SZ, 0, 0, SZ, SZ)
        val buf = FloatBuffer.allocate(3 * SZ * SZ)
        for (c in 0..2) for (p in px) {
            buf.put(when (c) { 0 -> (p shr 16) and 0xFF; 1 -> (p shr 8) and 0xFF; else -> p and 0xFF } / 255f)
        }
        buf.rewind()
        return OnnxTensor.createTensor(env, buf, longArrayOf(1, 3, SZ.toLong(), SZ.toLong()))
    }

    private fun parseOutput(raw: Any, h: Int, w: Int): List<DetectedObject> {
        val list = mutableListOf<DetectedObject>()
        val data = when (raw) {
            is Array<*> -> if (raw[0] is Array<*>) raw[0] as Array<*> else return list
            else -> return list
        }
        val isColMajor = data.size <= 84

        if (isColMajor) {
            val numBoxes = (data[0] as? FloatArray)?.size ?: return list
            val numCls   = data.size - 4
            for (i in 0 until numBoxes) {
                var best = 0f; var cls = 0
                for (c in 0 until numCls) {
                    val s = (data[4 + c] as FloatArray)[i]; if (s > best) { best = s; cls = c }
                }
                if (best < CONF) continue
                addBox(list, (data[0] as FloatArray)[i], (data[1] as FloatArray)[i],
                    (data[2] as FloatArray)[i], (data[3] as FloatArray)[i], best, cls, h, w)
            }
        } else {
            for (row in data) {
                row as FloatArray; if (row.size < 5) continue
                val scores = row.sliceArray(4 until row.size)
                val cls = scores.indices.maxByOrNull { scores[it] } ?: continue
                val conf = scores[cls]; if (conf < CONF) continue
                addBox(list, row[0], row[1], row[2], row[3], conf, cls, h, w)
            }
        }
        return nms(list).sortedBy { if (it.distanceM > 0) it.distanceM else 999f }.take(10)
    }

    private fun addBox(list: MutableList<DetectedObject>,
                       cx: Float, cy: Float, bw: Float, bh: Float,
                       conf: Float, cls: Int, origH: Int, origW: Int) {
        val x1    = ((cx - bw / 2f) / SZ).coerceIn(0f, 1f)
        val y1    = ((cy - bh / 2f) / SZ).coerceIn(0f, 1f)
        val x2    = ((cx + bw / 2f) / SZ).coerceIn(0f, 1f)
        val y2    = ((cy + bh / 2f) / SZ).coerceIn(0f, 1f)
        val label = if (cls < LABELS.size) LABELS[cls] else "object"
        val hPx   = (y2 - y1) * origH
        val dist  = HEIGHTS[label]?.let { rh -> if (hPx > 2f) (rh * FOCAL) / hPx else -1f } ?: -1f
        val midX  = (x1 + x2) / 2f
        val zone  = when { midX < 0.35f -> "left"; midX > 0.65f -> "right"; else -> "center" }
        list.add(DetectedObject(label, conf, RectF(x1, y1, x2, y2), dist, zone))
    }

    private fun nms(items: List<DetectedObject>): List<DetectedObject> {
        val sorted = items.sortedByDescending { it.confidence }
        val keep   = mutableListOf<DetectedObject>()
        val skip   = BooleanArray(sorted.size)
        for (i in sorted.indices) {
            if (skip[i]) continue; keep.add(sorted[i])
            for (j in i + 1 until sorted.size)
                if (!skip[j] && iou(sorted[i].box, sorted[j].box) > IOU) skip[j] = true
        }
        return keep
    }

    private fun iou(a: RectF, b: RectF): Float {
        val ix = maxOf(0f, minOf(a.right, b.right) - maxOf(a.left, b.left))
        val iy = maxOf(0f, minOf(a.bottom, b.bottom) - maxOf(a.top, b.top))
        val inter = ix * iy
        return inter / (a.width() * a.height() + b.width() * b.height() - inter + 1e-6f)
    }

    fun close() { session.close() }
}

// ═══════════════════════════════════════════════════════════════════════════
// SURFACE / WALL ANALYSER
// Detects walls, floors, ceilings without needing them as COCO classes.
// Strategy: analyse pixel brightness/edge density in frame zones.
// A wall close-up = uniform colour, fills whole zone, high homogeneity.
// An open path = varied depth cues, lower center coverage.
// ═══════════════════════════════════════════════════════════════════════════

object SurfaceAnalyser {

    // Returns 0.0 (fully open) to 1.0 (completely blocked) for each zone
    data class ZoneCoverage(
        val leftCoverage  : Float,
        val centerCoverage: Float,
        val rightCoverage : Float
    )

    fun analyse(bitmap: Bitmap): ZoneCoverage {
        val w  = bitmap.width
        val h  = bitmap.height

        // Focus on bottom 60% of frame — that's where the floor/wall boundary is
        val startY  = (h * 0.40f).toInt()
        val sampleH = h - startY

        val leftEnd   = (w * 0.35f).toInt()
        val centerEnd = (w * 0.65f).toInt()

        val leftCov   = zoneHomogeneity(bitmap, 0, leftEnd, startY, h)
        val centerCov = zoneHomogeneity(bitmap, leftEnd, centerEnd, startY, h)
        val rightCov  = zoneHomogeneity(bitmap, centerEnd, w, startY, h)

        return ZoneCoverage(leftCov, centerCov, rightCov)
    }

    // High homogeneity = uniform surface (wall/floor very close)
    // Low homogeneity  = varied scene (open space, objects at distance)
    private fun zoneHomogeneity(bmp: Bitmap, x1: Int, x2: Int, y1: Int, y2: Int): Float {
        if (x2 <= x1 || y2 <= y1) return 0f

        // Sample every 8th pixel for performance
        val step   = 8
        var rSum   = 0L; var gSum = 0L; var bSum = 0L
        var rSqSum = 0L; var gSqSum = 0L; var bSqSum = 0L
        var count  = 0

        var y = y1
        while (y < y2) {
            var x = x1
            while (x < x2) {
                val px = bmp.getPixel(x, y)
                val r  = (px shr 16) and 0xFF
                val g  = (px shr 8)  and 0xFF
                val b  = px          and 0xFF
                rSum  += r; gSum  += g; bSum  += b
                rSqSum += r * r; gSqSum += g * g; bSqSum += b * b
                count++
                x += step
            }
            y += step
        }

        if (count == 0) return 0f

        // Variance of each channel — low variance = homogeneous = wall/surface
        val rVar = (rSqSum.toFloat() / count) - (rSum.toFloat() / count).let { it * it }
        val gVar = (gSqSum.toFloat() / count) - (gSum.toFloat() / count).let { it * it }
        val bVar = (bSqSum.toFloat() / count) - (bSum.toFloat() / count).let { it * it }

        val avgVariance = (rVar + gVar + bVar) / 3f

        // Low variance (<400) = very homogeneous = close surface/wall
        // High variance (>2000) = lots of variety = open space
        // Map to 0-1 where 1 = blocked
        val coverage = (1f - (avgVariance / 1800f)).coerceIn(0f, 1f)
        return coverage
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PATH ANALYSER — combines YOLO + surface analysis
// ═══════════════════════════════════════════════════════════════════════════

object PathAnalyser {

    private const val DANGER  = 1.0f
    private const val WARNING = 2.5f

    // Wall detection thresholds — tuned for indoor use
    private const val WALL_VERY_CLOSE = 0.78f  // >78% homogeneous = wall right in front
    private const val WALL_CLOSE      = 0.55f  // >55% = wall getting close

    fun analyse(detections: List<DetectedObject>, bitmap: Bitmap): FrameAnalysis {
        val surface = SurfaceAnalyser.analyse(bitmap)

        // Check YOLO obstacles first (more precise than surface analysis)
        val centerObs = detections.filter { it.zone == "center" && it.label in Detector.OBSTACLES }
        val danger    = centerObs.firstOrNull { it.distanceM in 0.01f..DANGER }
        val warning   = centerObs.firstOrNull { it.distanceM in (DANGER + 0.01f)..WARNING }

        // Also check if a large object fills most of the center (table, wall, large furniture)
        val largeCenter = detections.filter { it.zone == "center" }.any { d ->
            val area = d.box.width() * d.box.height()
            area > 0.25f  // covers more than 25% of frame = large nearby object
        }

        val pathStatus = when {
            danger    != null                          -> PathStatus.OBSTACLE_DANGER
            warning   != null                          -> PathStatus.OBSTACLE_WARNING
            surface.centerCoverage > WALL_VERY_CLOSE  -> PathStatus.WALL_VERY_CLOSE
            largeCenter                                -> PathStatus.WALL_CLOSE
            surface.centerCoverage > WALL_CLOSE        -> PathStatus.WALL_CLOSE
            else                                       -> PathStatus.CLEAR
        }

        val leftOpen   = surface.leftCoverage   < WALL_CLOSE && detections.none { it.zone == "left"   && it.label in Detector.OBSTACLES }
        val rightOpen  = surface.rightCoverage  < WALL_CLOSE && detections.none { it.zone == "right"  && it.label in Detector.OBSTACLES }
        val centerOpen = pathStatus == PathStatus.CLEAR

        return FrameAnalysis(detections, pathStatus, surface.centerCoverage, leftOpen, rightOpen, centerOpen)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN ACTIVITY
// ═══════════════════════════════════════════════════════════════════════════

class MainActivity : AppCompatActivity(), TextToSpeech.OnInitListener {

    companion object {
        private const val TAG          = "BlindAssist"
        private const val REQ          = 42
        private val PERMS              = arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO)
        private const val NAV_INTERVAL = 2500L
        private const val DANGER       = 1.0f
        private const val WARNING      = 2.5f

        private val BOX_COLORS = mapOf(
            "person"       to Color.RED,
            "car"          to Color.YELLOW,
            "dining table" to Color.rgb(255, 165, 0),
            "chair"        to Color.rgb(255, 165, 0),
            "couch"        to Color.rgb(255, 165, 0),
            "dog"          to Color.MAGENTA,
            "cat"          to Color.MAGENTA,
            "bicycle"      to Color.YELLOW,
            "default"      to Color.WHITE
        )
    }

    // Views
    private lateinit var previewView   : PreviewView
    private lateinit var overlayView   : ImageView
    private lateinit var tvStatus      : TextView
    private lateinit var tvCommand     : TextView
    private lateinit var tvResponse    : TextView
    private lateinit var tvMic         : TextView

    // Services
    private lateinit var detector      : Detector
    private lateinit var tts           : TextToSpeech
    private lateinit var recognizer    : SpeechRecognizer
    private val executor: ExecutorService = Executors.newSingleThreadExecutor()
    private val handler = Handler(Looper.getMainLooper())

    // State
    private var ttsReady       = false
    private var speechOn       = false
    private var navOn          = false
    private var frameCount     = 0
    private var lastFrameAnalysis: FrameAnalysis? = null
    private var lastWarnTime   = 0L
    private val navRunnable    = Runnable { navCycle() }

    // ── Lifecycle ─────────────────────────────────────────────────────────

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContentView(R.layout.activity_main)

        previewView  = findViewById(R.id.previewView)
        overlayView  = findViewById(R.id.overlayView)
        tvStatus     = findViewById(R.id.tvStatus)
        tvCommand    = findViewById(R.id.tvCommand)
        tvResponse   = findViewById(R.id.tvResponse)
        tvMic        = findViewById(R.id.tvMic)

        tts = TextToSpeech(this, this)
        if (allGranted()) boot() else requestPerms()
    }

    override fun onDestroy() {
        super.onDestroy()
        speechOn = false; navOn = false
        handler.removeCallbacksAndMessages(null)
        if (::recognizer.isInitialized) recognizer.destroy()
        if (::detector.isInitialized)   detector.close()
        if (ttsReady) tts.shutdown()
        executor.shutdown()
    }

    override fun onRequestPermissionsResult(code: Int, p: Array<String>, g: IntArray) {
        super.onRequestPermissionsResult(code, p, g)
        if (code == REQ && allGranted()) boot()
        else ui { tvStatus.text = "Permissions required." }
    }

    // ── Boot ──────────────────────────────────────────────────────────────

    private fun boot() {
        ui { tvStatus.text = "Loading AI model..." }
        lifecycleScope.launch(Dispatchers.IO) {
            detector = Detector(this@MainActivity)
            withContext(Dispatchers.Main) {
                tvStatus.text = "Ready"
                startCamera()
                startSpeech()
                speak("Blind Assist is ready. I will warn you about walls and obstacles automatically. Say what do you see to begin.")
            }
        }
    }

    // ── TTS ───────────────────────────────────────────────────────────────

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            tts.language = Locale.US
            tts.setSpeechRate(0.90f)
            tts.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                override fun onStart(id: String?)  {}
                override fun onDone(id: String?)   {}
                @Deprecated("Deprecated in Java")
                override fun onError(id: String?)  {}
            })
            ttsReady = true
        }
    }

    private fun speak(text: String, flush: Boolean = true) {
        if (!ttsReady) return
        tts.speak(text, if (flush) TextToSpeech.QUEUE_FLUSH else TextToSpeech.QUEUE_ADD, null, UUID.randomUUID().toString())
        ui { tvResponse.text = text }
    }

    // ── Speech ────────────────────────────────────────────────────────────

    private fun startSpeech() {
        if (!SpeechRecognizer.isRecognitionAvailable(this)) {
            ui { tvMic.text = "Speech N/A" }; return
        }
        speechOn = true
        listenLoop()
    }

    private fun listenLoop() {
        if (!speechOn) return
        recognizer = SpeechRecognizer.createSpeechRecognizer(this)
        recognizer.setRecognitionListener(object : RecognitionListener {
            override fun onReadyForSpeech(p: Bundle?) { ui { tvMic.text = "Listening..." } }
            override fun onEndOfSpeech()              { ui { tvMic.text = "Processing..." } }
            override fun onResults(r: Bundle?) {
                val text = r?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                    ?.firstOrNull()?.trim()?.lowercase() ?: ""
                if (text.isNotEmpty()) {
                    ui { tvCommand.text = "You: \"$text\"" }
                    handleCommand(text)
                }
                handler.postDelayed({ listenLoop() }, 300)
            }
            override fun onError(e: Int) {
                val d = if (e == SpeechRecognizer.ERROR_RECOGNIZER_BUSY) 1000L else 400L
                if (speechOn) handler.postDelayed({ listenLoop() }, d)
            }
            override fun onBeginningOfSpeech()           {}
            override fun onRmsChanged(db: Float)         {}
            override fun onBufferReceived(b: ByteArray?) {}
            override fun onPartialResults(r: Bundle?)    {}
            override fun onEvent(t: Int, p: Bundle?)     {}
        })

        recognizer.startListening(  Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault())
            putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
            putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_COMPLETE_SILENCE_LENGTH_MILLIS, 1500L)
            putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_POSSIBLY_COMPLETE_SILENCE_LENGTH_MILLIS, 1500L)
        })
    }

    // ── Command Handler ───────────────────────────────────────────────────

    private fun handleCommand(cmd: String) {
        val fa = lastFrameAnalysis
        val objs = fa?.detections ?: emptyList()

        val reply = when {
            has(cmd, "what do you see","look","describe","around","see")
                -> cmdSee(fa)
            has(cmd, "path clear","safe","can i walk","is it clear","clear ahead","is path")
                -> cmdPathClear(fa)
            has(cmd, "someone in front","person ahead","anyone in front","is there a person","people")
                -> cmdPerson(objs)
            has(cmd, "on the left","my left","what's left","left side")
                -> cmdLeft(objs)
            has(cmd, "on the right","my right","what's right","right side")
                -> cmdRight(objs)
            has(cmd, "where should i go","which way","direction","where to go","navigate")
                -> cmdWhere(fa)
            has(cmd, "find exit","find door","find the door","way out","get out","exit","how do i get out","find the exit")
                -> cmdFindExit(fa)
            has(cmd, "how far","distance","how close","how many metres")
                -> cmdDistance(objs)
            has(cmd, "what is in front","what's in front","what's ahead","what is ahead")
                -> cmdAhead(fa)
            has(cmd, "start navigation","guide me","begin navigation","start guiding")
                -> { startNav(); "Navigation started. I will warn you automatically about walls and obstacles." }
            has(cmd, "stop navigation","stop guiding","stop guide")
                -> { stopNav(); "Navigation stopped." }
            has(cmd, "help","commands","what can you do","what can i say")
                -> helpText()
            else -> cmdSee(fa)
        }
        speak(reply)
    }

    // ═══════════════════════════════════════════════════════════════════════
    // COMMANDS — now all use FrameAnalysis for wall awareness
    // ═══════════════════════════════════════════════════════════════════════

    private fun cmdSee(fa: FrameAnalysis?): String {
        if (fa == null) return "Camera is starting up. Please wait."
        val sb = StringBuilder()

        // First describe path status
        when (fa.pathStatus) {
            PathStatus.WALL_VERY_CLOSE  -> sb.append("There is a wall or surface directly in front of you. ")
            PathStatus.WALL_CLOSE       -> sb.append("Something large is blocking the path ahead. ")
            PathStatus.OBSTACLE_DANGER  -> sb.append("Warning! Something is very close ahead. ")
            PathStatus.OBSTACLE_WARNING -> sb.append("Caution, something ahead. ")
            PathStatus.CLEAR            -> sb.append("The path ahead looks open. ")
        }

        // Then list specific objects
        if (fa.detections.isEmpty()) {
            sb.append("No specific objects identified.")
        } else {
            val parts = fa.detections.take(4).map { it.speak() }
            sb.append("I can see: ${parts.joinToString(". ")}.")
        }
        return sb.toString()
    }

    private fun cmdPathClear(fa: FrameAnalysis?): String {
        if (fa == null) return "Still initialising. Please wait."
        return when (fa.pathStatus) {
            PathStatus.CLEAR -> {
                val open = mutableListOf<String>()
                if (fa.leftOpen)   open.add("left")
                if (fa.centerOpen) open.add("center")
                if (fa.rightOpen)  open.add("right")
                "The path is clear. Open directions: ${open.joinToString(", ")}. You can move forward safely."
            }
            PathStatus.WALL_VERY_CLOSE -> {
                val turn = when {
                    fa.leftOpen && fa.rightOpen -> "You can turn either left or right."
                    fa.leftOpen  -> "Turn left to find a clear path."
                    fa.rightOpen -> "Turn right to find a clear path."
                    else         -> "All directions appear blocked. Stop and scan around slowly."
                }
                "Wall or surface directly in front of you! $turn"
            }
            PathStatus.WALL_CLOSE -> {
                val turn = when {
                    fa.leftOpen  -> "Try moving left."
                    fa.rightOpen -> "Try moving right."
                    else         -> "Slow down and stop."
                }
                "A large surface or object is close ahead. $turn"
            }
            PathStatus.OBSTACLE_DANGER -> {
                val obs = fa.detections.firstOrNull { it.zone == "center" && it.label in Detector.OBSTACLES }
                val name = obs?.label ?: "object"
                val dist = obs?.let { if (it.distanceM > 0f) " only ${fmt(it.distanceM)} metres away" else "" } ?: ""
                "Warning! $name$dist directly ahead. Stop immediately!"
            }
            PathStatus.OBSTACLE_WARNING -> {
                val obs = fa.detections.firstOrNull { it.zone == "center" && it.label in Detector.OBSTACLES }
                val name = obs?.label ?: "object"
                val dist = obs?.let { if (it.distanceM > 0f) " ${fmt(it.distanceM)} metres ahead" else " ahead" } ?: " ahead"
                "Caution. $name$dist. Slow down."
            }
        }
    }

    private fun cmdAhead(fa: FrameAnalysis?): String {
        if (fa == null) return "Still initialising."
        val centerObjs = fa.detections.filter { it.zone == "center" }

        return when {
            fa.pathStatus == PathStatus.WALL_VERY_CLOSE ->
                "There is a wall or solid surface directly in front of you."
            fa.pathStatus == PathStatus.WALL_CLOSE ->
                "There is a large object or surface blocking the way ahead."
            centerObjs.isNotEmpty() -> {
                val parts = centerObjs.take(3).joinToString(", ") {
                    val d = if (it.distanceM > 0f) " at ${fmt(it.distanceM)} metres" else ""
                    "${it.label}$d"
                }
                "In front of you: $parts."
            }
            else -> "Nothing detected directly ahead. The path looks open."
        }
    }

    private fun cmdPerson(objs: List<DetectedObject>): String {
        val p = objs.firstOrNull { it.label == "person" && it.zone == "center" }
            ?: return "No person detected directly in front of you."
        val d = if (p.distanceM > 0f) "about ${fmt(p.distanceM)} metres" else "nearby"
        return "Yes, there is a person in front of you, $d."
    }

    private fun cmdLeft(objs: List<DetectedObject>): String {
        val items = objs.filter { it.zone == "left" }
        if (items.isEmpty()) return "Nothing detected on your left side."
        return "On your left: ${items.take(3).joinToString(", ") { it.label }}."
    }

    private fun cmdRight(objs: List<DetectedObject>): String {
        val items = objs.filter { it.zone == "right" }
        if (items.isEmpty()) return "Nothing detected on your right side."
        return "On your right: ${items.take(3).joinToString(", ") { it.label }}."
    }

    private fun cmdWhere(fa: FrameAnalysis?): String {
        if (fa == null) return "Still initialising."

        // Check for exits first
        val exit = fa.detections.firstOrNull { it.label in Detector.EXIT_HINTS }
        if (exit != null) return buildExitDir(exit)

        return when {
            fa.centerOpen && !fa.leftOpen && !fa.rightOpen ->
                "The path straight ahead is clear. Move forward."
            fa.centerOpen ->
                "The path straight ahead is clear. Move forward."
            fa.leftOpen && fa.rightOpen ->
                "The path ahead is blocked. You can go either left or right."
            fa.leftOpen ->
                "Path ahead is blocked. Turn left — that direction is clear."
            fa.rightOpen ->
                "Path ahead is blocked. Turn right — that direction is clear."
            else ->
                "All directions appear blocked. Stop and turn around slowly to find an opening."
        }
    }

    private fun cmdFindExit(fa: FrameAnalysis?): String {
        if (fa == null) return "Still initialising."
        val objs = fa.detections

        // Look for explicit exit objects
        val exit = objs.firstOrNull { it.label in Detector.EXIT_HINTS }
        if (exit != null) return buildExitDir(exit)

        // Look for large rectangular objects that could be doors
        // A door is typically tall (height > width) and in a certain size range
        val doorCandidate = objs.firstOrNull { obj ->
            val h = obj.box.height()
            val w = obj.box.width()
            h > 0.3f && h > w * 1.3f  // tall and narrow = door-like shape
        }
        if (doorCandidate != null) {
            val dir = when (doorCandidate.zone) { "left" -> "Turn left"; "right" -> "Turn right"; else -> "Go straight" }
            return "I can see a tall opening that might be a door on your ${doorCandidate.zone}. $dir and move toward it."
        }

        // Use open path to guide toward exit
        return when {
            fa.leftOpen -> "No door visible. The left side is open — try turning left and scanning for an exit."
            fa.rightOpen -> "No door visible. The right side is open — try turning right and scanning for an exit."
            fa.centerOpen -> "No door visible ahead. Move forward slowly while I keep scanning for an exit."
            else -> "I cannot see an exit. Try turning around slowly — I will tell you when I spot a door or opening."
        }
    }

    private fun cmdDistance(objs: List<DetectedObject>): String {
        val known = objs.filter { it.distanceM > 0f }
        if (known.isEmpty()) return "Cannot estimate distances right now. Try moving closer to objects."
        return known.take(3).joinToString(". ") { "${it.label} is ${fmt(it.distanceM)} metres away" } + "."
    }

    private fun buildExitDir(e: DetectedObject): String {
        val dist = if (e.distanceM > 0f) "${fmt(e.distanceM)} metres" else "ahead"
        val dir  = when (e.zone) { "left" -> "Turn left"; "right" -> "Turn right"; else -> "Go straight" }
        return "Exit found! $dir toward the ${e.label}, $dist away."
    }

    private fun helpText() =
        "Commands: what do you see, is the path clear, what is in front, " +
                "is someone in front, left, right, where should I go, find exit, " +
                "how far, start navigation, stop navigation."

    // ── Navigation Mode ───────────────────────────────────────────────────

    private fun startNav() { navOn = true; handler.postDelayed(navRunnable, NAV_INTERVAL) }
    private fun stopNav()  { navOn = false; handler.removeCallbacks(navRunnable) }

    private fun navCycle() {
        if (!navOn) return
        val fa = lastFrameAnalysis
        if (fa != null) {
            val msg = when (fa.pathStatus) {
                PathStatus.WALL_VERY_CLOSE -> {
                    val turn = when { fa.leftOpen -> "Turn left."; fa.rightOpen -> "Turn right."; else -> "Stop." }
                    "Wall ahead! $turn"
                }
                PathStatus.WALL_CLOSE -> {
                    val turn = when { fa.leftOpen -> "Veer left."; fa.rightOpen -> "Veer right."; else -> "Slow down." }
                    "Surface close ahead. $turn"
                }
                PathStatus.OBSTACLE_DANGER -> {
                    val obs = fa.detections.firstOrNull { it.zone == "center" && it.label in Detector.OBSTACLES }
                    "Stop! ${obs?.label ?: "obstacle"} very close ahead!"
                }
                PathStatus.OBSTACLE_WARNING -> {
                    val obs = fa.detections.firstOrNull { it.zone == "center" && it.label in Detector.OBSTACLES }
                    "Careful. ${obs?.label ?: "obstacle"} ahead. Slow down."
                }
                PathStatus.CLEAR -> {
                    val exit = fa.detections.firstOrNull { it.label in Detector.EXIT_HINTS }
                    if (exit != null) buildExitDir(exit) else "Path clear. Continue moving."
                }
            }
            speak(msg, flush = false)
        }
        handler.postDelayed(navRunnable, NAV_INTERVAL)
    }

    // ── Camera ────────────────────────────────────────────────────────────

    private fun startCamera() {
        ProcessCameraProvider.getInstance(this).addListener({
            val provider = ProcessCameraProvider.getInstance(this).get()
            val preview  = Preview.Builder().build().also { it.setSurfaceProvider(previewView.surfaceProvider) }
            val analyzer = ImageAnalysis.Builder()
                .setTargetResolution(android.util.Size(640, 480))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build().also { ia -> ia.setAnalyzer(executor) { proxy -> analyzeFrame(proxy) } }
            provider.unbindAll()
            provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analyzer)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyzeFrame(proxy: ImageProxy) {
        frameCount++
        if (frameCount % 3 != 0) { proxy.close(); return }
        if (!::detector.isInitialized) { proxy.close(); return }

        val bmp  = proxy.toBitmap().rotate(proxy.imageInfo.rotationDegrees.toFloat())
        proxy.close()

        val dets = try { detector.run(bmp) } catch (e: Exception) {
            Log.e(TAG, "Detection error: ${e.message}"); return
        }

        // Run path + surface analysis
        val fa = PathAnalyser.analyse(dets, bmp)
        lastFrameAnalysis = fa

        // Update UI
        updatePathUI(fa)
        drawBoxes(bmp, dets, fa)

        // Auto-warn if wall very close (throttle to max 1 warn per 3 seconds)
        val now = System.currentTimeMillis()
        if (fa.pathStatus == PathStatus.WALL_VERY_CLOSE && now - lastWarnTime > 3000L) {
            lastWarnTime = now
            val turn = when { fa.leftOpen -> " Turn left."; fa.rightOpen -> " Turn right."; else -> "" }
            speak("Warning! Wall ahead.$turn", flush = true)
        }
    }

    // ── UI Updates ────────────────────────────────────────────────────────

    private fun updatePathUI(fa: FrameAnalysis) {
        ui {
            tvStatus.text = when (fa.pathStatus) {
                PathStatus.WALL_VERY_CLOSE -> "⛔ WALL AHEAD"
                PathStatus.WALL_CLOSE -> "⚠ Surface close"
                PathStatus.OBSTACLE_DANGER -> "⛔ OBSTACLE CLOSE"
                PathStatus.OBSTACLE_WARNING -> "⚠ Obstacle ahead"
                PathStatus.CLEAR -> "✓ Path clear"
            }
        }
    }

    // ── Bounding Box Drawing ──────────────────────────────────────────────

    private fun drawBoxes(bitmap: Bitmap, objs: List<DetectedObject>, fa: FrameAnalysis) {
        val w       = bitmap.width.toFloat()
        val h       = bitmap.height.toFloat()
        val overlay = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
        val canvas  = Canvas(overlay)

        val boxPaint = Paint().apply { style = Paint.Style.STROKE; strokeWidth = 5f; isAntiAlias = true }
        val bgPaint  = Paint().apply { style = Paint.Style.FILL;   isAntiAlias = true }
        val txtPaint = Paint().apply { color = Color.WHITE; textSize = 38f; isAntiAlias = true; isFakeBoldText = true }

        // Draw a subtle red overlay on center when wall detected
        if (fa.pathStatus == PathStatus.WALL_VERY_CLOSE || fa.pathStatus == PathStatus.WALL_CLOSE) {
            val warnPaint = Paint().apply {
                color = Color.argb(60, 255, 0, 0)
                style = Paint.Style.FILL
            }
            val cx1 = w * 0.35f; val cx2 = w * 0.65f
            canvas.drawRect(cx1, 0f, cx2, h, warnPaint)
        }

        for (obj in objs) {
            val color = BOX_COLORS[obj.label] ?: BOX_COLORS["default"]!!
            boxPaint.color = color
            bgPaint.color  = Color.argb(180, Color.red(color), Color.green(color), Color.blue(color))

            val l = obj.box.left   * w
            val t = obj.box.top    * h
            val r = obj.box.right  * w
            val b = obj.box.bottom * h

            canvas.drawRect(l, t, r, b, boxPaint)

            val lbl = if (obj.distanceM > 0f) "${obj.label} ${fmt(obj.distanceM)}m" else "${obj.label} ${(obj.confidence * 100).toInt()}%"
            val tw  = txtPaint.measureText(lbl) + 14f
            val th  = txtPaint.textSize + 8f
            val ly  = if (t > th) t else t + th + 4f

            canvas.drawRoundRect(l, ly - th, l + tw, ly + 4f, 8f, 8f, bgPaint)
            canvas.drawText(lbl, l + 7f, ly, txtPaint)
        }

        ui { overlayView.setImageBitmap(overlay) }
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    private fun has(input: String, vararg kw: String) = kw.any { it in input }
    private fun fmt(v: Float) = String.format("%.1f", v)
    private fun ui(block: () -> Unit) = runOnUiThread(block)
    private fun allGranted() = PERMS.all { ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED }
    private fun requestPerms() = ActivityCompat.requestPermissions(this, PERMS, REQ)
    private fun Bitmap.rotate(deg: Float): Bitmap {
        if (deg == 0f) return this
        return Bitmap.createBitmap(this, 0, 0, width, height, Matrix().apply { postRotate(deg) }, true)
    }
}
