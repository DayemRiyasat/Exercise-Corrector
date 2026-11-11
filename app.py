# app.py - ShapeForm web application
# Location: project_root/app.py
#
# All five exercises use the Exercise-Corrector pipeline in cv_v5/: the
# browser sends camera frames (or the uploaded clip), and the server runs
# YOLO11 pose and judges the movement with rules + a BiGRU. Squat, lunge and
# push-up are judged per finished rep, jump rope and plank per 2 second
# window.
#
# The original MediaPipe landmark route (/api/analyze_frame and friends,
# backed by exercises/) is still here but no exercise uses it any more.
#
# Structure:
#   Pages      -> marketing site, exercise library, analyser, dashboard
#   Analysis   -> /api/load_model, /api/analyze_frame, /api/analyze_image
#   Analysis (cv_v5 pipeline, every exercise)
#              -> /api/v5/live_frame, /api/v5/analyze_video, /api/v5/jobs/<id>,
#                 /api/v5/analyze_image
#   Sessions   -> /api/sessions (CRUD), /api/stats
#
# Predictors are imported lazily inside the factory below. Importing this
# module therefore does not pull in TensorFlow, which keeps page loads
# fast and means the site still serves if a model file is missing.

from datetime import datetime, timezone
from functools import wraps

from flask import (
    Flask, g, jsonify, make_response, redirect,
    render_template, request, url_for
)

import numpy as np

import content
import database
from cv_v5 import engine as cv5

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
# Uploaded clips are analysed on the server.
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024

DEVICE_COOKIE = 'shapeform_device'
COOKIE_MAX_AGE = 60 * 60 * 24 * 365 * 2   # two years

# Feedback copy is derived from the exercise library, so the coaching in
# the app and the guidance on the site can never drift apart.
FEEDBACK_MESSAGES = content.feedback_messages()


# ---------------------------------------------------------------------------
# Device identity
# ---------------------------------------------------------------------------
# There are no accounts. Each browser gets an anonymous ID in a cookie and
# their sessions hang off that. It is the smallest thing that makes history
# work across page loads without asking anyone to sign up.

@app.before_request
def attach_device_id():
    g.device_id = request.cookies.get(DEVICE_COOKIE) or database.new_device_id()
    g.device_is_new = request.cookies.get(DEVICE_COOKIE) != g.device_id


@app.after_request
def persist_device_id(response):
    if getattr(g, 'device_is_new', False):
        response.set_cookie(
            DEVICE_COOKIE, g.device_id,
            max_age=COOKIE_MAX_AGE, samesite='Lax', httponly=True
        )
    return response


@app.context_processor
def inject_globals():
    """Everything the shared layout needs on every page."""
    return {
        'site_stats': content.site_stats(),
        'nav_exercises': content.all_exercises(),
        'safety_note': content.SAFETY_NOTE,
        'current_year': datetime.now(timezone.utc).year,
        'active_page': request.endpoint,
    }


def json_endpoint(fn):
    """Turn an unhandled error into JSON rather than an HTML error page,
    so the front end always gets something it can parse."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:                      # noqa: BLE001
            app.logger.exception('API error in %s', fn.__name__)
            return jsonify({'success': False, 'error': str(exc)}), 500
    return wrapper


# ---------------------------------------------------------------------------
# Landmark plumbing
# ---------------------------------------------------------------------------
class _Landmark:
    """Lightweight stand-in for a MediaPipe landmark, so the existing
    predictor and rep-counter code can access .x / .y / .z / .visibility
    without any changes."""
    __slots__ = ('x', 'y', 'z', 'visibility')

    def __init__(self, d):
        self.x = float(d.get('x', 0.0))
        self.y = float(d.get('y', 0.0))
        self.z = float(d.get('z', 0.0))
        self.visibility = float(d.get('visibility', 0.0))


def _wrap_landmarks(raw):
    """Turn the JSON array from the browser into landmark objects.
    Returns None when no usable pose was sent."""
    if not raw or len(raw) < 33:
        return None
    return [_Landmark(d) for d in raw]


def get_predictor_for_exercise(exercise_type):
    """Factory for exercise predictors.

    The imports are deliberately inside the function. TensorFlow is only
    pulled in when someone actually starts an analysis, so serving the
    marketing pages and the library costs nothing.
    """
    if exercise_type == 'squat':
        from exercises.squat.SquatPredictor import SquatPredictor
        return SquatPredictor()
    if exercise_type == 'lunge':
        from exercises.lunge.LungePredictor import LungePredictor
        return LungePredictor()
    if exercise_type == 'pushup':
        from exercises.pushup.PushupPredictor import PushupPredictor
        return PushupPredictor()
    return None


# One predictor per exercise, cached inside each worker process.
_predictors = {}


def _get_or_load(exercise_type):
    if not exercise_type or exercise_type not in content.EXERCISES:
        return None
    predictor = _predictors.get(exercise_type)
    if predictor is None:
        try:
            predictor = get_predictor_for_exercise(exercise_type)
        except Exception:                              # noqa: BLE001
            app.logger.exception('Could not construct predictor for %s', exercise_type)
            return None
        if predictor is None or not predictor.load_model():
            return None
        _predictors[exercise_type] = predictor
    return predictor


def _is_v5(exercise_type):
    return content.EXERCISES.get(exercise_type, {}).get('engine') == 'v5'


_live_sessions = cv5.LiveSessions()
_video_jobs = cv5.VideoJobs()


def _v5_engine_or_error():
    """The v5 engine, or a JSON error the analyser can show when torch /
    ultralytics are missing or the bundle will not load."""
    try:
        return cv5.get_engine(), None
    except Exception as exc:                           # noqa: BLE001
        app.logger.exception('Could not load the v5 pipeline')
        return None, jsonify({'success': False,
                              'error': f'The v5 pipeline could not load: {exc}'})


def _v5_feedback(exercise_type, verdict):
    """Coaching for one judged rep or window (or photo): the strongest
    fault's copy, or the clean copy, with weaker 'hint' detections added as tips."""
    messages = FEEDBACK_MESSAGES.get(exercise_type, {})
    faults = verdict.get('faults') or []
    key = faults[0]['key'] if faults else 'none'
    fb = dict(messages.get(key, messages.get('unknown', {})))
    fb['tips'] = list(fb.get('tips', []))
    for f in faults[1:]:
        extra = messages.get(f['key'])
        if extra:
            fb['tips'].insert(0, extra['message'])
    for h in verdict.get('hints') or []:
        extra = messages.get(h['key'])
        if extra:
            fb['tips'].append('Watch: ' + extra['message'].lower())
    fb['key'] = key
    return fb


def _read_image(file_storage, max_side=640):
    """Uploaded JPEG/PNG -> BGR array, long side <= max_side (the size the
    pipeline decodes video at). None if it is not an image."""
    import cv2
    buf = np.frombuffer(file_storage.read(), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None
    if img is None:
        return None
    h, w = img.shape[:2]
    s = max_side / float(max(h, w))
    if s < 1:
        img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    return img


def _attach_feedback(result, exercise_type):
    if result.get('success'):
        messages = FEEDBACK_MESSAGES.get(exercise_type, {})
        feedback = messages.get(result['prediction'], messages.get('unknown', {}))
        result['feedback'] = dict(feedback)
    return result


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template(
        'index.html',
        how_it_works=content.HOW_IT_WORKS,
        features=content.FEATURES,
        exercises=content.all_exercises(),
        faq=content.FAQ[:5],
    )


@app.route('/analyze')
def analyze():
    return render_template('analyze.html', exercises=content.all_exercises())


@app.route('/exercises')
def exercise_library():
    return render_template('exercises.html', exercises=content.all_exercises())


@app.route('/exercises/<slug>')
def exercise_detail(slug):
    exercise = content.get_exercise(slug)
    if not exercise:
        return render_template('404.html'), 404

    faults = [
        {**fault, 'key': key}
        for key, fault in exercise['faults'].items()
        if key not in content.NON_FAULT_KEYS
    ]
    clean = exercise['faults'].get('none')

    others = [e for e in content.all_exercises() if e['slug'] != slug][:3]
    return render_template(
        'exercise_detail.html',
        exercise=exercise, faults=faults, clean=clean, others=others
    )


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html', exercises=content.all_exercises())


@app.route('/faq')
def faq():
    return render_template('faq.html', faq=content.FAQ)


@app.route('/about')
def about():
    return render_template('about.html', features=content.FEATURES)


@app.route('/privacy')
def privacy():
    return render_template('privacy.html')


# Old bookmarks pointed at the tool on the site root.
@app.route('/index.html')
def legacy_index():
    return redirect(url_for('analyze'), code=301)


# ---------------------------------------------------------------------------
# Analysis API
# ---------------------------------------------------------------------------
@app.route('/api/load_model', methods=['POST'])
@json_endpoint
def load_model():
    data = request.get_json(silent=True) or {}
    exercise_type = data.get('exercise_type')

    if _is_v5(exercise_type):
        _, error = _v5_engine_or_error()
        if error is not None:
            return error
        _live_sessions.reset(g.device_id, exercise_type)
        return jsonify({
            'success': True,
            'engine': 'v5',
            'message': f'{content.EXERCISES[exercise_type]["name"]} pipeline ready',
            'exercise': exercise_type
        })

    predictor = _get_or_load(exercise_type)
    if predictor is None:
        return jsonify({'success': False,
                        'error': f'Could not load model for "{exercise_type}"'})

    predictor.reset_counter()
    return jsonify({
        'success': True,
        'message': f'{content.EXERCISES[exercise_type]["name"]} model ready',
        'exercise': exercise_type
    })


@app.route('/api/analyze_frame', methods=['POST'])
@json_endpoint
def analyze_frame():
    """Webcam and video frames. Browser sends landmarks, server predicts
    and advances the rep counter."""
    data = request.get_json(silent=True) or {}
    exercise_type = data.get('exercise_type')
    landmarks = _wrap_landmarks(data.get('landmarks'))

    predictor = _get_or_load(exercise_type)
    if predictor is None:
        return jsonify({'success': False, 'error': 'No model loaded'})
    if landmarks is None:
        return jsonify({'success': False, 'error': 'No pose detected',
                        'message': 'Position yourself in frame'})

    result = predictor.analyze_landmarks(landmarks, count_rep=True)
    return jsonify(_attach_feedback(result, exercise_type))


@app.route('/api/analyze_image', methods=['POST'])
@json_endpoint
def analyze_image():
    """Single still image. Browser sends landmarks, server predicts and
    returns the detailed analysis. Rep counter is not advanced."""
    data = request.get_json(silent=True) or {}
    exercise_type = data.get('exercise_type')
    landmarks = _wrap_landmarks(data.get('landmarks'))

    predictor = _get_or_load(exercise_type)
    if predictor is None:
        return jsonify({'success': False, 'error': 'No model loaded'})
    if landmarks is None:
        return jsonify({'success': False, 'error': 'No pose detected',
                        'message': 'Ensure full body is visible in the image'})

    result = predictor.analyze_image_landmarks(landmarks)
    return jsonify(_attach_feedback(result, exercise_type))


@app.route('/api/reset_counter', methods=['POST'])
@json_endpoint
def reset_counter():
    data = request.get_json(silent=True) or {}
    exercise_type = data.get('exercise_type')
    if _is_v5(exercise_type):
        _live_sessions.reset(g.device_id, exercise_type)
        return jsonify({'success': True, 'message': 'Rep counter reset', 'rep_count': 0})
    predictor = _predictors.get(exercise_type) if exercise_type else None

    if predictor is None:
        return jsonify({'success': True, 'message': 'Nothing to reset', 'rep_count': 0})

    predictor.reset_counter()
    return jsonify({'success': True, 'message': 'Rep counter reset', 'rep_count': 0})


# ---------------------------------------------------------------------------
# Analysis API, cv_v5 pipeline (every exercise)
# ---------------------------------------------------------------------------
def _v5_exercise_from_form():
    exercise_type = request.form.get('exercise_type')
    return exercise_type if _is_v5(exercise_type) else None


@app.route('/api/v5/live_frame', methods=['POST'])
@json_endpoint
def v5_live_frame():
    """One webcam frame (JPEG) plus its capture time in ms. Returns the
    skeleton for this frame and any rep the counter has just closed."""
    exercise_type = _v5_exercise_from_form()
    if not exercise_type:
        return jsonify({'success': False, 'error': 'Not a v5 exercise'}), 400
    engine, error = _v5_engine_or_error()
    if error is not None:
        return error
    img = _read_image(request.files['frame']) if 'frame' in request.files else None
    if img is None:
        return jsonify({'success': False, 'error': 'No frame received'}), 400
    try:
        t_ms = float(request.form.get('t', '0'))
    except ValueError:
        t_ms = 0.0

    live = _live_sessions.get(g.device_id, exercise_type)
    result = live.add_frame(img, t_ms)
    for rep in result['new_reps']:
        rep['feedback'] = _v5_feedback(exercise_type, rep)
    return jsonify(result)


@app.route('/api/v5/analyze_video', methods=['POST'])
@json_endpoint
def v5_analyze_video():
    """Start analysing an uploaded clip. Poll /api/v5/jobs/<id> for it."""
    exercise_type = _v5_exercise_from_form()
    if not exercise_type:
        return jsonify({'success': False, 'error': 'Not a v5 exercise'}), 400
    if 'video' not in request.files:
        return jsonify({'success': False, 'error': 'No video received'}), 400
    _, error = _v5_engine_or_error()
    if error is not None:
        return error
    job = _video_jobs.start(request.files['video'], exercise_type, g.device_id)
    return jsonify({'success': True, 'job_id': job['id']})


@app.route('/api/v5/jobs/<job_id>', methods=['GET'])
@json_endpoint
def v5_job_status(job_id):
    job = _video_jobs.get(job_id, g.device_id)
    if job is None:
        return jsonify({'success': False, 'error': 'Job not found'}), 404
    out = {'success': True, 'status': job['status'], 'progress': job['progress'],
           'message': job['message']}
    if job['status'] in ('done', 'error'):
        result = job['result'] or {'success': False, 'error': job['message']}
        for rep in result.get('reps', []):
            rep['feedback'] = _v5_feedback(job['exercise'], rep)
        out['result'] = result
    return jsonify(out)


@app.route('/api/v5/analyze_image', methods=['POST'])
@json_endpoint
def v5_analyze_image():
    """One photo: pose plus the rule checks that work without movement."""
    exercise_type = _v5_exercise_from_form()
    if not exercise_type:
        return jsonify({'success': False, 'error': 'Not a v5 exercise'}), 400
    engine, error = _v5_engine_or_error()
    if error is not None:
        return error
    img = _read_image(request.files['image']) if 'image' in request.files else None
    if img is None:
        return jsonify({'success': False, 'error': 'Could not read that image'}), 400
    result = engine.analyze_image(img, exercise_type)
    if result.get('success'):
        result['feedback'] = _v5_feedback(exercise_type, result)
    return jsonify(result)


# ---------------------------------------------------------------------------
# Sessions API
# ---------------------------------------------------------------------------
@app.route('/api/sessions', methods=['GET'])
@json_endpoint
def list_sessions():
    exercise = request.args.get('exercise')
    sessions = database.list_sessions(g.device_id, exercise=exercise)
    return jsonify({'success': True, 'sessions': sessions, 'count': len(sessions)})


@app.route('/api/sessions', methods=['POST'])
@json_endpoint
def create_session():
    data = request.get_json(silent=True) or {}

    if data.get('exercise') not in content.EXERCISES:
        return jsonify({'success': False, 'error': 'Unknown exercise'}), 400
    if not int(data.get('reps') or 0) > 0:
        return jsonify({'success': False, 'error': 'A session needs at least one rep'}), 400

    session = database.create_session(g.device_id, data)
    return jsonify({'success': True, 'session': session}), 201


@app.route('/api/sessions/<session_id>', methods=['DELETE'])
@json_endpoint
def remove_session(session_id):
    ok = database.delete_session(g.device_id, session_id)
    if not ok:
        return jsonify({'success': False, 'error': 'Session not found'}), 404
    return jsonify({'success': True})


@app.route('/api/sessions', methods=['DELETE'])
@json_endpoint
def remove_all_sessions():
    removed = database.clear_sessions(g.device_id)
    return jsonify({'success': True, 'removed': removed})


@app.route('/api/stats', methods=['GET'])
@json_endpoint
def stats():
    data = database.compute_stats(g.device_id)
    # Attach display names so the front end does not need its own copy
    # of the exercise library.
    data['labels'] = {
        slug: {'name': ex['name'], 'icon': ex['icon'], 'unit': ex.get('unit', 'rep')}
        for slug, ex in content.EXERCISES.items()
    }
    # Retired exercises keep their name in saved sessions.
    data['labels'].update(content.LEGACY_EXERCISES)
    data['faultLabels'] = dict(content.LEGACY_FAULT_LABELS)
    data['faultLabels'].update({
        key: content.fault_label(slug, key)
        for slug in content.EXERCISES
        for key in content.EXERCISES[slug]['faults']
    })
    return jsonify({'success': True, 'stats': data})


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'loaded_exercises': list(_predictors.keys()),
        'available_exercises': content.EXERCISE_ORDER,
        'v5_exercises': [s for s in content.EXERCISE_ORDER if _is_v5(s)],
        'v5_pipeline_loaded': cv5._engine is not None,
    })


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
@app.errorhandler(404)
def not_found(_e):
    if request.path.startswith('/api/'):
        return jsonify({'success': False, 'error': 'Not found'}), 404
    return render_template('404.html'), 404


@app.errorhandler(500)
def server_error(_e):
    if request.path.startswith('/api/'):
        return jsonify({'success': False, 'error': 'Server error'}), 500
    return render_template('500.html'), 500


# Build the schema at import time so the WSGI server picks it up too.
database.init_db()


if __name__ == '__main__':
    # Local testing only. PythonAnywhere uses the WSGI file, not this block.
    print('Starting Flask server. Visit: http://localhost:5000')
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
