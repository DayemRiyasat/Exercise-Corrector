# Exercise-Corrector v5.1: the new pipeline for squat, lunge and push-up

This branch replaces the computer vision behind **squat, lunge and push-up**
with the Exercise-Corrector v5.1 pipeline. On the `v6-jump-rope-plank` branch,
**jump rope and plank replace deadlift and bicep curl**, so every exercise now
runs on this pipeline and MediaPipe is no longer used. See
[v6: jump rope and plank](#v6-jump-rope-and-plank).

This document covers what changed, how the new pipeline works, what it was
trained on, how well it does, and how to run it. The main [README](README.md)
still describes the rest of the app.

---

## What changed, in short

| | Before | Now (squat, lunge, push-up) |
|---|---|---|
| Pose model | MediaPipe in the browser, 33 points | YOLO11x-pose on the server, 17 points |
| What gets judged | Every single frame, on its own | Every **whole rep**, from start to finish |
| How it judges | One small neural network per exercise | Rules on joint angles **plus** a per-rep BiGRU network, combined |
| Training data | Single poses (one frame at a time) | ~11,300 real exercise **videos** from QEVD, plus 3D motion capture seen from 12 camera angles |
| Camera position | Side-on only | Front, side or diagonal; the pipeline works out the view itself |
| Verdict | A label per frame | Per rep: clean or named faults, a score out of 100, and minor "watch" hints |
| Tested on | Not reported | People the model never saw in training (see the results below) |

---

## v6: jump rope and plank

Deadlift and bicep curl are gone from the site. Jump rope and elbow plank take
their place, trained on QEVD (`air jump rope`, `elbow plank`) on the same
pipeline. Squat, lunge and push-up behave exactly as before: the same clip
gives the same verdicts on both branches.

| | Jump rope | Plank |
|---|---|---|
| What gets judged | Every **2 second window** (32 frames at 16 fps), because jumps are too quick and small to cut one by one | Every **2 second window**, because a hold has no reps |
| Faults | Feet not leaving the floor, heels only, side to side, no wrist circles | Hips sagging, hips high, head down, head up, on knees, shoulders shrugged, elbows too wide |
| Counter on screen | Jumps (display only, the verdict comes from the windows) | Seconds held (time with a person in frame) |
| Camera | Face the camera, phone at waist height, feet in frame | Film front-on, camera at floor level, 2 to 3 m in front, whole body visible. Side-on filming is not supported yet |
| Photo | Not offered: one photo cannot show it | Rule checks only, like the other exercises |

A saved session stores the number of windows judged in its rep count, and the
dashboard labels those sessions "Windows".

**Results** on QEVD test people the models never saw, per clip, after the v6.1
calibration. A clip counts as faulty if any of its 2 second windows is.

| | Test clips (clean) | Clean called clean | Faulty caught | Balanced accuracy |
|---|---|---|---|---|
| Jump rope | 206 (54) | 96% | 97% | 0.97 |
| Plank | 577 (54) | 85% | 69% | 0.77 |

Before calibration the plank scored 0.50, because every clean plank was called
faulty.

| Fault | F1 | | Fault | F1 |
|---|---|---|---|---|
| Jump rope: heels only | 1.00 | | Plank: head down | 0.77 |
| Jump rope: side to side | 0.99 | | Plank: head up | 0.67 |
| Jump rope: no wrist circles | 0.97 | | Plank: hips sagging | 0.60 |
| Jump rope: no jump | 0.95 | | Plank: elbows too wide | 0.59 |
| | | | Plank: shoulders shrugged | 0.53 |
| | | | Plank: hips high | 0.51 |
| | | | Plank: on knees | 0.48 |

Plank faults are noticeably weaker than jump rope faults, so treat them as
hints rather than diagnoses. The level one "clean or not" call is the stronger
signal.

Known limits, from testing on real clips outside QEVD:

* **Plank is reliable front-on only.** About 96% of the plank data is filmed
  front-on. Two real clips of reasonable planks filmed side-on and diagonally
  were flagged as on knees and hips sagging in every window, so the numbers
  above do not cover those views yet. The site tells users to film front-on.
* **Plank photos use hand-set rule thresholds** that were not fitted on data,
  so a photo verdict is a rough guide.
* **Jump counts are approximate.** On one clip the upload route counted 18
  jumps, live at 16 fps counted 16, and live at 4.4 fps (a laptop CPU) counted
  13. They are for display only and do not affect the verdicts.

The calibration is in `calibration_v51.json` next to the v5.1 values, which are
unchanged. The deadlift and bicep curl models have been removed. The
`exercises/` folder keeps only the old squat, lunge and push-up MediaPipe models,
which sit behind the old `/api/analyze_frame` route that no page calls any more.

---

## How the pipeline works

```
webcam frames / uploaded video / photo
        │
        ▼
YOLO11x-pose (server)  ──►  17 joints per frame, resampled to 16 fps
        │
        ▼
clean-up: fill short gaps, One Euro smoothing
        │
        ▼
rep counter: finds where each rep starts, bottoms out and ends
        │
        ▼  for every finished rep
 ┌──────────────────────────────┬────────────────────────────────────────┐
 │ Judge 1: rules               │ Judge 2: per-rep BiGRU                 │
 │ depth, knee gap, stance,     │ the whole rep (32 frames x 80 features │
 │ trunk lean, knee past ankle, │ of joints, angles and velocities)      │
 │ hip sag / pike, lockout,     │ ─► "is this rep clean?"                │
 │ back-knee angle; each rule   │ ─► "which faults are present?"         │
 │ only votes when the camera   │                                        │
 │ view lets it see the fault   │                                        │
 └──────────────┬───────────────┴───────────────────┬────────────────────┘
                └──────────── weighted fusion ──────┘
                                 │
                                 ▼
        verdict: clean, or the faults found + score 0-100 + coaching text
```

The rep counter and the judges are separate on purpose: a rep is always
counted, then graded. That keeps the fix described in the main README.

### Faults it names

| Exercise | Faults |
|---|---|
| Squat | too shallow, knees caving in, leaning forward, knees too far forward, stance too wide, stance too narrow |
| Lunge | too shallow, front knee too far forward, leaning forward, back knee too straight |
| Push-up | too shallow, hips sagging, hips too high, head dropping, no lockout |

If the network is sure a rep is not clean but no single fault stands out, the
rep is marked **"Form differs from a clean rep"** (`form_issue`).

### The three ways to use it in the app

* **Upload a video:** the server runs the exact pipeline that was evaluated on
  Kaggle as a background job, with a progress bar. The clip then plays back
  with the skeleton, and each rep's verdict appears as the rep finishes.
* **Webcam:** each frame is sent to the server as soon as the previous one is
  answered (about 4 to 6 frames a second on a laptop CPU with YOLO11x). The
  frames are put back on the pipeline's 16 fps clock, and every rep is judged
  the moment it ends.
* **Photo:** a single photo has no movement, so only the rules that work on a
  still pose run (stance, knees caving in, lean, hips). The page lists which
  checks it could not do and suggests a video for them.

---

## The dataset

### QEVD (main training data)

The **Qualcomm Exercise Video Dataset (QEVD-FIT-300k)**: short real videos of
people exercising at home in front of a phone or webcam, each tagged with
form attributes such as "shallow", "knees over toes", "leaning forward" or
"no obvious issue". All four parts of the dataset were downloaded and every
squat, push-up and lunge clip was used:

| Exercise | Train | Validation | Test | Total clips |
|---|---|---|---|---|
| Squat | 2,863 | 572 | 385 | 3,820 |
| Push-up | 2,344 | 475 | 314 | 3,133 |
| Lunge | 3,333 | 641 | 421 | 4,395 |
| **All** | **8,540** | **1,688** | **1,120** | **11,348** |

How the labels were made trustworthy:

* **Split by person (75 / 15 / 10).** Nobody in the test set appears in
  training, so the test numbers show how the model does on new people, not
  on faces it has memorised.
* **Every QEVD tag was given one role**: a named fault, a sign of good form
  (e.g. "90 degrees", "shoulder-width"), a general issue, something irrelevant
  to form (tempo, "hands behind head"), or "camera can't see". Clips where the
  person is not fully visible were dropped.
* **Unknown labels are masked, not guessed.** If a clip says nothing about a
  fault, that fault contributes nothing to training for that clip.
* **Poor-quality clips were filtered out**: shorter than 1 second, nobody
  found in more than 30% of frames, or low joint confidence.

### EC3D (more camera angles)

QEVD is filmed mostly from the front. To teach the model side and diagonal
views, the **EC3D** dataset (3D motion capture of squats and lunges with
labelled mistakes, ACCV 2022) was used. Every rep was projected into
**12 camera angles** (front, side, back and diagonals, with small random
jitter). Two known problems in EC3D were corrected first: its torso is forced
upright in every frame (which hides forward lean), and its left and right are
mirrored.

---

## Improvements in camera angles

* **Before:** the exercise guides told users to film side-on only, and the
  old models judged single still poses from that angle.
* **Now:** the BiGRU has seen reps from front, side, back and diagonal views
  (real front-facing QEVD video plus EC3D from 12 angles). It works from
  wherever a phone is likely to be propped up.
* **The pipeline detects the view itself** (front, side or diagonal) from the
  body's proportions, and each rule only votes when that view can show its
  fault. For example, knees caving in needs a front view and torso lean needs
  a side or diagonal view. When a rule cannot see, the network decides alone.
* The exercise guides now say: front or slight angle for squats and lunges,
  side or slight angle at floor level for push-ups, front-on at floor level
  for planks.

## Improvements in dataset quality

* **Real video instead of still poses:** over 11,000 clips of real people in
  real rooms, with real lighting and clothing, filmed on phones and webcams.
  That is the same kind of footage the app will see.
* **Whole reps instead of frames:** faults like "too shallow" or "no lockout"
  only exist across a movement. A single frame can't show them.
* **QEVD's form tags mapped carefully** onto one fixed fault list per
  exercise, with uncertain labels masked rather than forced.
* **An honest test:** all numbers below are measured on people the model never
  trained on.

---

## Results (v5.1, test people never seen in training)

**Clean vs. not clean, per clip:**

| Exercise | Clean reps called clean | Faulty reps caught | Balanced accuracy |
|---|---|---|---|
| Squat | 83% | 76% | **0.80** |
| Lunge | 91% | 79% | **0.85** |
| Push-up | 29% | 99% | 0.64 |

**Per fault (F1 score, 1.0 is perfect):**

| Squat | F1 | Lunge | F1 | Push-up | F1 |
|---|---|---|---|---|---|
| too shallow | 0.81 | too shallow | 0.69 | too shallow | 0.78 |
| stance too narrow | 0.70 | back knee straight | 0.78 | head dropping | 0.80 |
| leaning forward | 0.66 | leaning forward | 0.76 | hips too high | 0.68 |
| stance too wide | 0.57 | front knee forward | 0.61 | hips sagging | 0.29 |
| knees too far forward | 0.45 | | | | |

Honest limits:

* **Push-up "clean" calls are weak.** QEVD has very few clean push-ups (69 out
  of 3,133 clips), so the push-up verdict comes from the named faults rather
  than the "clean?" head. Expect push-ups to be flagged more often than they
  deserve.
* **Hips sagging (push-up) and knees too far forward (squat)** are the least
  reliable faults.
* **Knees caving in (squat) and no lockout (push-up)** have no QEVD test
  examples, so they rely on the rules (and, for knees caving in, on EC3D).
* YOLO's 17 joints have no feet, so "knee past toes" is measured as knee past
  ankle, and heel lift is not checked.
* Webcam frames reach the server at about 4 to 6 a second rather than the 16
  a second the model was evaluated at. Rep counts held up in testing; scores
  can differ slightly from an uploaded video of the same set.

---

## Running it

You need **Python 3.11** (TensorFlow and MediaPipe pin it. They are only used
by the retired models in `exercises/`, but they are still in `requirements.txt`).

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# CPU build of torch first, so pip doesn't pull the 2 GB CUDA build
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python app.py                      # then open http://localhost:5000
```

* **YOLO weights:** `yolo11x-pose.pt` (118 MB) is too large for git. The first
  time an exercise is opened, ultralytics downloads it
  automatically into `cv_v5/` (internet needed once; it can take a couple of
  minutes, so that first request is slow). To skip it, put a copy in `cv_v5/`.
* The first analysis takes a few seconds longer while the models load.
* Optional environment variables:
  * `EC_LIVE_POSE_WEIGHTS=yolo11s-pose.pt` uses a smaller, much faster pose
    model for the webcam only. It wasn't part of the evaluation.
  * `EC_MAX_VIDEO_SECONDS` caps how much of an upload is analysed (default 180).
  * `EC_DEVICE=cuda` runs on a GPU if one is available.

### Hosting

The new pipeline runs on the server, so the host needs room for torch,
ultralytics and the 118 MB pose model, and enough CPU for the webcam mode.
**PythonAnywhere's free tier is not enough.** A small VPS or a paid
PythonAnywhere plan with more disk and CPU is needed, ideally with a GPU for
many simultaneous webcam users. Video analysis runs in a background thread,
so the host must allow threads.

---

## Notes for the client

* **Privacy wording changed.** The site used to say video never leaves the
  device. That's no longer true for any exercise: frames, clips and photos
  are sent to the server, analysed and discarded (never stored).
  The privacy page, FAQ, home page, about page and footer were reworded to
  say so. Please review the wording.
* **Coaching text** now follows the new fault list (`content.py`). Sessions
  saved with the old fault names, including old deadlift and bicep curl
  sessions, still show readable labels on the dashboard.
* **Licences:** QEVD is released for **research use only**, and EC3D is an
  academic dataset. Check the licence terms before any commercial use of
  models trained on them.

---

## Where things are in the code

```
cv_v5/
├── ec_pipeline.py          The v6 pipeline, exactly as trained and evaluated (unchanged)
├── engine.py               Connects it to the app: video jobs, live webcam sessions, photos
├── pipeline_config.json    Fitted rule thresholds, fault list, coaching text, metrics
├── calibration_v51.json    v5.1 + v6.1 cut-offs and judge weights (tuned on validation people)
└── models/<exercise>/pose_gru.pt   The five BiGRUs

app.py                      /api/v5/live_frame, /api/v5/analyze_video,
                            /api/v5/jobs/<id>, /api/v5/analyze_image
static/js/v5_client.js      Sends frames / clips / photos, draws the 17-joint skeleton
static/js/app.js            Uses the v5 engine for exercises marked engine: 'v5'
content.py                  'engine': 'v5' on every exercise, 'unit': 'window' on jump rope
                            and plank, plus their faults
```
