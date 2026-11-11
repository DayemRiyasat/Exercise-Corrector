# Exercise-Corrector

Point a camera at yourself while you lift, and this tells you what your form is
doing wrong while you are still moving.

It counts your reps for you, names the specific thing you are getting wrong, and
explains how to fix it. Everything the camera sees stays on your own device.

Five exercises, sixteen different mistakes it can spot, and a dashboard that
tracks whether you are actually getting better.

## Why I made this

Bad form is really hard to notice from the inside. You cannot see your own back
angle halfway through a squat, and by the time you watch a video back, the set
is finished and you have already done fifteen reps wrong.

The thing is, a lot of what a coach spots is just shapes. Knees caving inwards.
Hips shooting up before the shoulders. Hips sagging in a plank. A
camera can see those shapes too, and it can tell you about them in under a
second, which is early enough to actually change the next rep.

That is the whole idea. It is not trying to replace a coach. It just closes the
gap between doing the rep and finding out you did it wrong.

I built this as my project for TKS Demo Day.

## What it does

* Watches your form live through a webcam, or from a video you upload, or from a
  single photo
* Counts your reps on its own, and only counts one when you actually complete
  the full movement
* Tells you the specific mistake instead of giving you a vague score, and
  explains what that mistake costs you and how to fix it
* Saves every session, then shows your rep counts over time, how many were
  clean, your training streak, and which mistakes keep coming back
* Includes a written guide for each exercise covering where to put the camera,
  how to set up, and every mistake it looks for
* Works without an account. Your history is tied to your browser, not a login
* Has a light and a dark theme, and remembers which one you picked

## How it works

This section describes the original MediaPipe route, which no exercise uses any
more. What runs today is in the next section.

The short version: your browser figures out where your body is, sends just the
joint positions to the server, and the server decides whether the movement looks
right.

In more detail:

1. Your browser takes a frame from the camera and works out the position of 33
   points on your body, things like your shoulders, hips, knees and ankles. This
   happens on your own machine, inside the page, using a Google library called
   MediaPipe.
2. Only those 33 positions get sent to the server. They are just numbers. The
   actual picture is thrown away and never leaves your device.
3. The server runs those numbers through a small neural network trained for the
   exercise you picked. It answers one question: does this look like good form,
   or does it look like one of the specific mistakes it was trained on?
4. Separately, the server tracks where you are in the movement, standing, going
   down, at the bottom, coming back up, and adds one to your count when you
   finish a full cycle.
5. The answer gets matched to the coaching text and drawn on screen, about seven
   times a second.

One design decision worth calling out: counting a rep and judging a rep are kept
completely separate. A rep still counts even if your form was bad, and gets
marked as clean or not afterwards. That sounds obvious, but the first version
threw away any rep with a mistake in it, which meant the counter stopped working
at exactly the moment you needed the feedback most.

## All five exercises: the v5.1 / v6 pipeline

Full details (architecture, dataset, results, camera angles, hosting) are in
[README_V5.md](README_V5.md).

No exercise uses the MediaPipe models above any more. The `exercises/` folder
now only holds the old squat, lunge and push-up models, which no page calls any
more, and deadlift and bicep curl are gone from
the site. Every exercise runs the Exercise-Corrector pipeline in `cv_v5/`, which
judges whole reps instead of single frames. Jump rope and plank have no reps, so
they are judged in fixed 2 second windows instead:

```
camera frames or uploaded clip ─► YOLO11x-pose (17 joints, on the server, 16 fps)
   ─► fill gaps + One Euro smoothing ─► rep counter (start, bottom, end of each rep)
   ─► for every finished rep:
        rules on joint angles (depth, knees, stance, lean, hips, lockout)
      + a per-rep BiGRU per exercise (is the rep clean? which faults?)
      ─► fused verdict: clean or named faults, plus a score out of 100
```

* `cv_v5/ec_pipeline.py` is the pipeline exactly as trained and evaluated on
  Kaggle, copied unchanged. `calibration_v51.json` holds the calibrated
  thresholds, `models/<exercise>/pose_gru.pt` the five BiGRUs.
* `cv_v5/engine.py` connects it to the web app:
  * **Uploaded video** goes through the exact Kaggle code path as a
    background job; the page then plays the clip back with the skeleton and
    shows each rep's verdict as the rep finishes.
  * **Webcam** frames are sent one at a time (as fast as the server answers,
    about 4 to 6 a second on a laptop CPU). They are put back on the
    pipeline's 16 fps clock, and each rep is judged the moment the rep
    counter closes it.
  * **A photo** can only use the rule checks that do not need movement
    (stance, knees caving in, lean, hips). Depth and the BiGRU need a full rep.
    Jump rope cannot be judged from a photo, so the option is hidden for it.
* The YOLO weights (`yolo11x-pose.pt`, 118 MB) are not in git. Put a copy in
  `cv_v5/`, or let ultralytics download it on first use.
* Optional settings: `EC_LIVE_POSE_WEIGHTS=yolo11s-pose.pt` uses a smaller,
  faster pose model for the webcam only (not part of the evaluation);
  `EC_MAX_VIDEO_SECONDS` caps how much of an upload is analysed (default 180);
  `EC_DEVICE=cuda` uses a GPU.

The fault names for these exercises in `content.py` are the pipeline's own
list, plus `form_issue` for a rep the model is sure is not clean without one
fault standing out. Sessions saved with the old names still show readable
labels on the dashboard.

## The exercises

| Exercise | Difficulty | Mistakes it spots | Where to put the camera |
|---|---|---|---|
| Squat | Beginner | 6 | In front or at a slight angle, about hip height |
| Lunge | Beginner | 4 | In front or at a slight angle |
| Push-up | Beginner | 5 | To your side or at a slight angle, down near the floor |
| Jump Rope | Beginner | 4 | In front of you, waist height, feet in frame |
| Plank | Beginner | 7 | In front of you at floor level, 2 to 3 m away, whole body visible. Side-on is not supported yet |

Each model can also say "that looked fine"
or "I cannot see enough of you to tell".

All of this lives in one file, `content.py`. Every mistake is written down once,
with both the short message that appears on screen mid-set and the longer
explanation that appears in the guide. The website and the app read from the
same place, so they cannot end up saying different things.

## What is in the project

```
Exercise-Corrector/
├── app.py                  The web server: pages and the API
├── content.py              All the exercise writing and coaching text
├── database.py             Saving and summarising your sessions
├── base_predictor.py       Shared logic for loading models and judging reps
├── requirements.txt
│
├── exercises/
│   └── squat/ lunge/ pushup/      (old MediaPipe route, no page calls these)
│       ├── ...Predictor.py      Turns joint positions into numbers to classify
│       ├── RepCounter.py        Tracks where you are in the movement
│       └── models/              The trained model and its settings files
│
├── templates/              The pages, as HTML
└── static/
    ├── style.css           Every bit of the design
    ├── favicon.svg         Plus .ico and apple-touch-icon
    └── js/
        ├── v5_client.js    Sends frames, clips and photos to the server
        ├── pose_client.js  The old MediaPipe route, no longer loaded
        ├── app.js          Runs the analyser page
        ├── dashboard.js    Draws the charts and history
        ├── api.js          Small helper for calling the server
        └── site.js         Menu, theme switch, FAQ, scroll effects
```

## Running it on your own machine

You need Python 3.11. The code itself would run on older versions, but the
machine learning libraries are what pin it.

```bash
git clone https://github.com/<your-username>/Exercise-Corrector.git
cd Exercise-Corrector

python3.11 -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python app.py
```

Then go to http://localhost:5000.

The database file makes itself the first time you run it, so there is no setup
step.

One thing that catches people out: browsers only let a site use your camera on
`localhost` or on a proper HTTPS address. If you open the site using your
computer's local network address instead, the camera will silently never ask for
permission.

## Putting it online with PythonAnywhere

1. Upload the project, or clone it from a console.
2. Make a virtual environment on Python 3.11 and install the requirements:

   ```bash
   mkvirtualenv --python=/usr/bin/python3.11 exercise-corrector
   pip install -r requirements.txt
   ```

3. On the Web tab, point it at your project folder and at the virtual
   environment you just made, then edit the WSGI file so it reads:

   ```python
   import sys
   path = '/home/<username>/Exercise-Corrector'
   if path not in sys.path:
       sys.path.insert(0, path)

   from app import app as application
   ```

4. Add a static files mapping from `/static/` to your project's `static` folder.
5. Reload the web app.

Two things I learned the hard way here.

TensorFlow is a big install and free accounts do not get much space. If the
install stops halfway through, have a look at the tidying up section below,
because a few of the listed requirements are not actually used any more.

Also, file paths have to be absolute. When PythonAnywhere runs your site it does
not start from your project folder, so anything written as a relative path will
not be found. Every model loaded fine on my laptop and none of them loaded on
the server until this was fixed.

## The pages

| Address | What it is |
|---|---|
| `/` | Home page |
| `/analyze` | The actual tool |
| `/exercises` | List of the exercise guides |
| `/exercises/<name>` | The full guide for one exercise |
| `/dashboard` | Your history and progress charts |
| `/faq`, `/about`, `/privacy` | Supporting pages |

## The API

If you want to build something on top of this, everything comes back as JSON.

**Running an analysis**

| Method | Address | What it does |
|---|---|---|
| POST | `/api/load_model` | Gets an exercise ready |
| POST | `/api/analyze_frame` | Judges one frame and updates the rep count |
| POST | `/api/analyze_image` | Judges a single photo in more detail |
| POST | `/api/reset_counter` | Sets the rep count back to zero |

**Squat, lunge and push-up (v5 pipeline)**, sent as `multipart/form-data` with
`exercise_type`:

| Method | Address | What it does |
|---|---|---|
| POST | `/api/v5/live_frame` | One webcam JPEG (`frame`) and its time in ms (`t`); returns the skeleton and any rep just judged |
| POST | `/api/v5/analyze_video` | Uploads a clip (`video`) and starts a job |
| GET | `/api/v5/jobs/<id>` | The job's progress, then every rep's verdict and the skeleton for each frame |
| POST | `/api/v5/analyze_image` | One photo (`image`), judged with the rule checks |

**Your sessions**

| Method | Address | What it does |
|---|---|---|
| GET | `/api/sessions` | Your saved sessions, newest first |
| POST | `/api/sessions` | Saves a finished set |
| DELETE | `/api/sessions/<id>` | Deletes one session |
| DELETE | `/api/sessions` | Clears your whole history |
| GET | `/api/stats` | Your totals, streak and most common mistakes |
| GET | `/api/health` | Quick check that the server is alive |

Here is what comes back while you are mid-set:

```jsonc
{
  "success": true,
  "prediction": "extreme_forward_lean",
  "confidence": 0.91,
  "rep_info": {
    "rep_count": 7,
    "rep_counted": true,
    "rep_clean": false,
    "dominant_fault": "extreme_forward_lean"
  },
  "feedback": {
    "status": "FORWARD LEAN",
    "message": "Do not lean too far forward",
    "tips": ["Keep chest up", "Sit back into the squat"],
    "color": "warning"
  }
}
```

## What gets stored, and what does not

The webcam frames, uploaded video or photo are
sent to the server, because the pose model runs there. Frames and photos are
analysed in memory and dropped; an uploaded video sits in a temporary file
while it is analysed and is deleted straight after. None of it is stored.

When you finish a set the server saves how long it took,
how many reps you did, how many were clean, and which mistakes came up.

There are no accounts, no email addresses and no passwords. The first time you
visit, the server gives your browser a random ID and attaches your sessions to
it. It identifies a browser, not a person. Clear your cookies and you start
over. There is also a button on the dashboard that deletes everything.

Sessions are capped at 500 per browser, and the oldest get dropped after that.

## About the design

Black and white with one dark blue accent, which turned out to need two
different versions of that blue. The dark blue that looks right on a white card
completely disappears on a near black header, so there is a lighter version for
dark backgrounds.

The rest is fairly plain on purpose. Sharp corners, thin dividing lines instead
of drop shadows, Space Grotesk for headings and Inter for reading text. Numbers
use a setting that keeps every digit the same width, so the rep counter does not
jiggle while it counts.

Every colour is defined in one block at the top of `style.css`, which is also
how the dark theme works. It swaps that block out and everything else follows.

## If something is not working

**The reps are not counting.** Almost certainly the form check. There is a
setting that used to throw away any rep with a mistake in it, which meant the
counter froze exactly when you were getting feedback. Counting and grading need
to stay separate.

**My squats are not counting but everything else is.** The squat used to only
accept a very deep squat, deeper than most people actually go. The depth setting
in `exercises/squat/RepCounter.py` needs loosening so a normal parallel squat
counts.

**The skeleton is drawn slightly off my body.** The video and the drawing layer
on top of it need to be sized the same way. If one is stretched to fill the box
and the other is letterboxed, the skeleton drifts away from your arms and legs
towards the edges of the frame.

**The models will not load on the server.** Almost always relative file paths.
The loader prints out exactly which files it went looking for and where, so the
error message should tell you.

**The camera never asks for permission.** You are on a plain http address.
Browsers only allow camera access on localhost or HTTPS.

## Tidying up

There are two files in here that nothing uses any more: `static/js/script.js`
and `video_processor.py`. Both are left over from an earlier version and can be
deleted.

Once `video_processor.py` is gone, these can come out of `requirements.txt`,
because the work they used to do now happens elsewhere:

```
mediapipe, tqdm, pillow
```

With `exercises/` unused, tensorflow, joblib and scikit-learn can go too. Keep
opencv-python, which the `cv_v5/` pipeline needs. Worth double checking that
nothing in your model training still needs the others first.

## What it cannot do

This gives you general feedback about your form from a camera. It is not medical
advice.

* It only knows what the camera can see. If you are badly framed it will give
  you confident but wrong answers rather than no answer.
* It knows nothing about pain, past injuries, or how your particular body is
  built.
* It judges the shape you are making, not what you meant to do. If you are doing
  a variation on purpose, it may read that as a mistake.
* It cannot see how heavy the weight is. Perfect form with far too much weight
  still looks perfect to it.

If something hurts, stop, and go talk to an actual coach or physio.

## Credits

Body tracking by [YOLO11](https://docs.ultralytics.com/tasks/pose/).
The models run on [PyTorch](https://pytorch.org/).
Icons from [Bootstrap Icons](https://icons.getbootstrap.com/).
