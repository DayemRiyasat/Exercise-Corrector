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
Hips shooting up before the shoulders. An elbow drifting forward on a curl. A
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

## The exercises

| Exercise | Difficulty | Mistakes it spots | Where to put the camera |
|---|---|---|---|
| Squat | Beginner | 4 | To your side, about hip height |
| Lunge | Beginner | 4 | To your side |
| Push-up | Beginner | 3 | To your side, down near the floor |
| Deadlift | Intermediate | 3 | To your side, about hip height |
| Bicep curl | Beginner | 6 | In front of you |

Sixteen distinct mistakes in total. Each model can also say "that looked fine"
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
│   └── squat/ lunge/ pushup/ deadlift/ bicep_curl/
│       ├── ...Predictor.py      Turns joint positions into numbers to classify
│       ├── RepCounter.py        Tracks where you are in the movement
│       └── models/              The trained model and its settings files
│
├── templates/              The pages, as HTML
└── static/
    ├── style.css           Every bit of the design
    ├── favicon.svg         Plus .ico and apple-touch-icon
    └── js/
        ├── pose_client.js  Talks to MediaPipe and to the server
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

Never leaves your device: your webcam feed, any video you upload, any photo you
upload.

Sent to the server: 33 joint positions per frame, which are just numbers, plus
which exercise you chose. When you finish a set it also saves how long it took,
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
because the work they used to do now happens in the browser:

```
opencv-python, mediapipe, tqdm, pillow
```

Which leaves flask, werkzeug, tensorflow, numpy, joblib and scikit-learn. Worth
double checking that nothing in your model training still needs them first.

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

Body tracking by [MediaPipe](https://ai.google.dev/edge/mediapipe).
The models run on [TensorFlow](https://www.tensorflow.org/).
Icons from [Bootstrap Icons](https://icons.getbootstrap.com/).
