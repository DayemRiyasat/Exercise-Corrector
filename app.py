# app.py - ShapeForm web application
# Location: project_root/app.py
#
# The browser runs MediaPipe pose detection and posts the 33 landmarks to
# the analysis endpoints. The server runs the trained classification model
# on those landmarks. No server-side mediapipe, opencv, or background
# threading, so this runs on PythonAnywhere.
#
# Structure:
#   Pages      -> marketing site, exercise library, analyser, dashboard
#   Analysis   -> /api/load_model, /api/analyze_frame, /api/analyze_image
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

import content
import database

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

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
    if exercise_type == 'deadlift':
        from exercises.deadlift.DeadliftPredictor import DeadliftPredictor
        return DeadliftPredictor()
    if exercise_type == 'bicep_curl':
        from exercises.bicep_curl.BicepCurlPredictor import BicepCurlPredictor
        return BicepCurlPredictor()
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
    predictor = _predictors.get(exercise_type) if exercise_type else None

    if predictor is None:
        return jsonify({'success': True, 'message': 'Nothing to reset', 'rep_count': 0})

    predictor.reset_counter()
    return jsonify({'success': True, 'message': 'Rep counter reset', 'rep_count': 0})


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
        slug: {'name': ex['name'], 'icon': ex['icon']}
        for slug, ex in content.EXERCISES.items()
    }
    data['faultLabels'] = {
        key: content.fault_label(slug, key)
        for slug in content.EXERCISES
        for key in content.EXERCISES[slug]['faults']
    }
    return jsonify({'success': True, 'stats': data})


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'loaded_exercises': list(_predictors.keys()),
        'available_exercises': content.EXERCISE_ORDER,
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
