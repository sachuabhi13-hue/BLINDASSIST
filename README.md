It is an AI-powered navigation assistant for blind people. The user points their phone camera at the world, speaks a command, and the app speaks back describing what it sees, warns about obstacles, and guides them safely.

The Big Picture
REAL WORLD
    ↓ (camera)
PHONE PROCESSES EVERYTHING
    ├── YOLOv8 AI → finds objects
    ├── Surface Analyser → detects walls
    ├── Navigation Engine → understands commands
    ├── Speech Recognition → hears your voice
    └── Text-to-Speech → speaks back
    ↓
BLIND USER HEARS GUIDANCE

No internet. No server. Everything runs on the phone.

The 6 Core Components
1. YOLOv8 Detector (Detector class)

What it is: The AI brain that identifies objects in the camera frame.

How it works:

Camera Frame (e.g. 1920×1080)
    ↓ resize
640×640 pixels
    ↓ normalize pixels 0.0–1.0
    ↓ reshape to [1, 3, 640, 640]
    ↓ feed into yolov8n.onnx
Output: 8400 possible detections
    ↓ filter by confidence > 40%
    ↓ Non-Maximum Suppression (remove duplicates)
Final: 5–10 clean detections

What YOLO detects: 80 standard objects — person, chair, car, dining table, bottle, laptop, dog, cat, door, stairs, etc.

Distance estimation:

Distance = (Real height of object × Focal length) / Box height in pixels

Example:
Person real height = 1.70 metres
Focal length = 580 pixels
Box height = 300 pixels

Distance = (1.70 × 580) / 300 = 3.3 metres

Zone detection:

Camera frame split into 3 zones:

0%      35%      65%     100%
├─ LEFT ─┼─ CENTER ─┼─ RIGHT ─┤

Object center at 20% → LEFT zone
Object center at 50% → CENTER zone
Object center at 80% → RIGHT zone
2. Surface Analyser (SurfaceAnalyser object)

What it is: The wall/floor/ceiling detector. This is the fix for the problem you reported — YOLOv8 cannot detect walls because walls are not in the COCO dataset. So we use a completely different technique.

The problem it solves:

Before: User walks toward wall
        YOLO sees nothing (wall not a COCO class)
        App says "Path clear" ← WRONG AND DANGEROUS

After:  User walks toward wall
        Surface Analyser detects homogeneous surface
        App says "Wall ahead! Turn left." ← CORRECT

How it works — Pixel Variance Analysis:

Take camera frame
    ↓
Focus on bottom 60% (where floor/wall boundary is)
    ↓
Sample every 8th pixel for speed
    ↓
Calculate colour variance in each zone

LOW variance  = all pixels similar colour
             = uniform surface = WALL very close

HIGH variance = many different colours
             = varied scene = open space, objects at distance

Real example:

Pointing at open room:
    Pixels: sky blue, furniture brown, floor grey...
    Variance = HIGH → Coverage = 0.2 → Path CLEAR ✅

Pointing at white wall 20cm away:
    Pixels: white, white, white, white...
    Variance = VERY LOW → Coverage = 0.9 → WALL AHEAD ⛔

Coverage thresholds:

Coverage > 0.78 → Wall very close (STOP)
Coverage > 0.55 → Surface close (slow down)
Coverage < 0.55 → Path open
3. Path Analyser (PathAnalyser object)

What it is: The decision maker. Combines YOLO detections AND surface analysis to give a final verdict on the path.

Priority order:

1. YOLO obstacle within 1 metre  → OBSTACLE_DANGER  (highest priority)
2. YOLO obstacle within 2.5m     → OBSTACLE_WARNING
3. Surface coverage > 78%        → WALL_VERY_CLOSE
4. Large object fills 25%+ frame → WALL_CLOSE
5. Surface coverage > 55%        → WALL_CLOSE
6. None of the above             → CLEAR

Why this order matters: A person standing 0.5m away is more urgent than a wall 0.5m away. YOLO is more precise for known objects so it takes priority.

Zone openness:

For each zone (left, center, right):
    Open = surface coverage < 55%
           AND no YOLO obstacle in that zone

This tells the user WHERE to go when blocked:
"Path blocked. Turn left — that direction is clear."
4. Speech Recognition (SpeechService / listenLoop)

What it is: The always-listening voice input system.

How always-listening works:

App starts
    ↓
Start SpeechRecognizer
    ↓
Listen for speech
    ↓
User speaks → result received
    ↓
Process command
    ↓
Wait 300ms
    ↓
Restart SpeechRecognizer automatically ← always listening loop
    ↓
Listen again...

Why restart instead of keep open: Android's SpeechRecognizer automatically stops after detecting silence. We restart it immediately so there's no gap.

Command matching:

User says: "is the path clear"
    ↓
App checks: does the text contain "path clear"? YES
    ↓
Calls: cmdPathClear()
    ↓
Returns response based on current FrameAnalysis

Every command checks multiple variations:

kotlin
has(cmd, "find exit","find door","find the door",
         "way out","get out","exit","how do i get out")

So whether the user says "find exit" or "way out" or "get out" — it all works.

5. Text-to-Speech (TTSEngine)

What it is: Android's built-in voice that speaks responses out loud.

How it works:

Response text generated
    ↓
tts.speak(text, QUEUE_FLUSH, ...)
    ↓
Android TTS engine converts to audio
    ↓
Plays through phone speaker

Two modes:

QUEUE_FLUSH → interrupts anything currently speaking (used for warnings)
QUEUE_ADD   → waits its turn (used for navigation updates)

Why QUEUE_FLUSH for warnings: If the app is describing objects and suddenly detects a wall — it stops talking and immediately says "WALL AHEAD!"

6. Navigation Mode

What it is: Automatic continuous guidance every 2.5 seconds without needing voice commands.

How it works:

User says "Start navigation"
    ↓
navOn = true
    ↓
Every 2.5 seconds:
    Look at latest FrameAnalysis
        ↓
    WALL_VERY_CLOSE → "Wall ahead! Turn left."
    WALL_CLOSE      → "Surface close. Veer right."
    OBSTACLE_DANGER → "Stop! Person very close ahead!"
    OBSTACLE_WARNING→ "Careful. Chair 1.8 metres ahead."
    CLEAR           → "Path clear. Continue moving."
        ↓
    Speak the message
        ↓
    Wait 2.5 seconds
        ↓
    Repeat

Auto wall warning (even without navigation mode):
If a wall is detected at any time — even without navigation mode — the app automatically speaks a warning. This is the safety net for blind users who might forget to say "start navigation".

Data Flow — One Complete Cycle
┌─────────────────────────────────────────────────────┐
│  EVERY ~100ms (10 frames per second)                │
│                                                     │
│  Camera Frame                                       │
│      ↓                                              │
│  detector.run(bitmap)                               │
│      ↓                                              │
│  List<DetectedObject>  ← YOLOv8 results            │
│      ↓                                              │
│  SurfaceAnalyser.analyse(bitmap)                    │
│      ↓                                              │
│  ZoneCoverage  ← wall/surface analysis             │
│      ↓                                              │
│  PathAnalyser.analyse(detections, bitmap)           │
│      ↓                                              │
│  FrameAnalysis {                                    │
│      detections: [person 2m center, chair 1m left] │
│      pathStatus: OBSTACLE_WARNING                   │
│      blockCoverage: 0.42                            │
│      leftOpen: false                                │
│      rightOpen: true                                │
│      centerOpen: false                              │
│  }                                                  │
│      ↓                                              │
│  updatePathUI()  ← update screen colours           │
│  drawBoxes()     ← draw rectangles on camera       │
│  autoWarn()      ← speak if wall detected          │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  WHEN USER SPEAKS                                   │
│                                                     │
│  "Is the path clear?"                               │
│      ↓                                              │
│  SpeechRecognizer → text = "is the path clear"     │
│      ↓                                              │
│  handleCommand("is the path clear")                 │
│      ↓                                              │
│  cmdPathClear(lastFrameAnalysis)                    │
│      ↓                                              │
│  Reads FrameAnalysis                                │
│  pathStatus = WALL_VERY_CLOSE                       │
│      ↓                                              │
│  Returns: "Wall directly in front! Turn right,     │
│            that direction is clear."                │
│      ↓                                              │
│  tts.speak(response)  ← user hears this            │
└─────────────────────────────────────────────────────┘
File Structure Explained
BlindAssist/
│
├── app/build.gradle
│   → Lists all libraries the app needs
│   → Tells Android what phone version to support
│   → Sets Java 17 compatibility
│
├── app/src/main/AndroidManifest.xml
│   → Declares permissions (camera, microphone)
│   → Registers MainActivity as the launch screen
│
├── app/src/main/assets/yolov8n.onnx
│   → The AI model file (12 MB)
│   → All 80 COCO object classes baked in
│   → Runs entirely on phone CPU
│
├── app/src/main/java/com/blindassist/MainActivity.kt
│   → THE ENTIRE APP in one file (~600 lines)
│   → Contains 6 classes:
│       DetectedObject  — data model for one detection
│       FrameAnalysis   — result of full frame analysis
│       Detector        — YOLOv8 ONNX inference
│       SurfaceAnalyser — wall/floor detection
│       PathAnalyser    — combines both into verdict
│       MainActivity    — UI + speech + camera + TTS
│
└── app/src/main/res/
    ├── layout/activity_main.xml  → screen design
    ├── values/themes.xml         → app colour theme
    ├── values/strings.xml        → text strings
    └── values/colors.xml         → colour definitions
Why Certain Decisions Were Made
Decision	Why
YOLOv8 Nano (yolov8n)	Fastest model, runs at 10fps on phone CPU
ONNX format	Works on Android without Python/PyTorch
Pixel variance for walls	Walls have no COCO class — need different approach
Always-listening speech	Blind users cannot look at screen to tap a button
QUEUE_FLUSH for warnings	Safety — wall warning must interrupt everything
Process every 3rd frame	Saves battery and CPU — 10fps is enough for walking speed
Java 17	Required by newer Android Gradle Plugin
AppCompat theme	More compatible than MaterialComponents across Android versions
