# app.py - Landmark-based main application
# Location: project_root/app.py
#
# The browser runs MediaPipe pose detection and posts the 33 landmarks
# to these endpoints. The server runs the trained classification model
# on those landmarks. No server-side mediapipe, opencv, or background
# threading, so this runs on PythonAnywhere.

from flask import Flask, render_template, request, jsonify

from exercises.squat.SquatPredictor import SquatPredictor
from exercises.lunge.LungePredictor import LungePredictor
from exercises.pushup.PushupPredictor import PushupPredictor
from exercises.deadlift.DeadliftPredictor import DeadliftPredictor
from exercises.bicep_curl.BicepCurlPredictor import BicepCurlPredictor

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Feedback messages configuration
FEEDBACK_MESSAGES = {
    'squat': {
        'none': {'status': 'EXCELLENT FORM ✓', 'message': 'Perfect squat form!', 'tips': ['Keep back straight', 'Chest up', 'Knees aligned with toes', 'Great work!'], 'color': 'success'},
        'extreme_backward_lean': {'status': 'BACK ISSUE ⚠', 'message': 'Do not lean backward', 'tips': ['Engage core muscles', 'Maintain neutral spine', 'Keep weight centered'], 'color': 'danger'},
        'extreme_forward_lean': {'status': 'FORWARD LEAN ⚠', 'message': 'Do not lean too far forward', 'tips': ['Keep chest up', 'Look forward', 'Sit back into the squat'], 'color': 'warning'},
        'foots_too_close': {'status': 'STANCE TOO NARROW ⚠', 'message': 'Widen your stance', 'tips': ['Feet shoulder-width apart', 'Toes slightly pointed out'], 'color': 'warning'},
        'foots_too_far': {'status': 'STANCE TOO WIDE ⚠', 'message': 'Narrow your stance', 'tips': ['Bring feet closer', 'Maintain control'], 'color': 'warning'},
        'unknown': {'status': 'UNCLEAR POSITION', 'message': 'Position yourself in frame', 'tips': ['Ensure full body visible', 'Stand in good lighting'], 'color': 'secondary'}
    },
    'lunge': {
        'none': {'status': 'EXCELLENT FORM ✓', 'message': 'Perfect lunge form!', 'tips': ['Keep back straight', 'Front knee over ankle', 'Back knee bent', 'Great work!'], 'color': 'success'},
        'extreme_backward_lean': {'status': 'BACK ISSUE ⚠', 'message': 'Do not lean backward', 'tips': ['Engage core muscles', 'Keep torso upright', 'Look forward'], 'color': 'danger'},
        'extreme_forward_lean': {'status': 'FORWARD LEAN ⚠', 'message': 'Do not lean too far forward', 'tips': ['Keep chest up', 'Shoulders back', 'Stay vertical'], 'color': 'warning'},
        'foots_too_close': {'status': 'STANCE TOO NARROW ⚠', 'message': 'Step further forward', 'tips': ['Increase stride length', 'Front foot should be forward'], 'color': 'warning'},
        'foots_too_far': {'status': 'STANCE TOO WIDE ⚠', 'message': 'Reduce stride length', 'tips': ['Step closer', 'Maintain balance'], 'color': 'warning'},
        'unknown': {'status': 'UNCLEAR POSITION', 'message': 'Position yourself in frame', 'tips': ['Ensure full body visible', 'Stand in good lighting'], 'color': 'secondary'}
    },
    'pushup': {
        'none': {'status': 'EXCELLENT FORM ✓', 'message': 'Perfect push-up form!', 'tips': ['Elbows close to body', 'Straight back', 'Full range of motion', 'Great work!'], 'color': 'success'},
        'hand_too_far_or_incorrect_position': {'status': 'HAND POSITION ⚠', 'message': 'Adjust hand placement', 'tips': ['Hands shoulder-width apart', 'Position under shoulders', 'Fingers forward'], 'color': 'warning'},
        'hips_too_high': {'status': 'HIP POSITION ⚠', 'message': 'Lower your hips', 'tips': ['Maintain plank position', 'Keep core engaged', 'Straight line head to heels'], 'color': 'warning'},
        'incorrect_leg_position': {'status': 'LEG ALIGNMENT ⚠', 'message': 'Check leg position', 'tips': ['Keep legs straight', 'Feet together', 'Toes on ground'], 'color': 'warning'},
        'unknown': {'status': 'UNCLEAR POSITION', 'message': 'Position yourself in frame', 'tips': ['Ensure full body visible', 'Stand in good lighting'], 'color': 'secondary'}
    },
    'deadlift': {
        'none': {'status': 'EXCELLENT FORM ✓', 'message': 'Perfect deadlift form!', 'tips': ['Neutral spine', 'Chest up', 'Hips and shoulders rise together', 'Great work!'], 'color': 'success'},
        'back_arch_posture': {'status': 'BACK ARCH - CRITICAL ⚠', 'message': 'Keep spine neutral!', 'tips': ['Engage core', 'Chest up', 'Do not hyperextend back', 'Maintain neutral spine throughout'], 'color': 'danger'},
        'hand_grip_width': {'status': 'GRIP WIDTH ⚠', 'message': 'Adjust hand position', 'tips': ['Hands shoulder-width or slightly wider', 'Arms straight', 'Grip outside knees'], 'color': 'warning'},
        'leg_position_width': {'status': 'STANCE WIDTH ⚠', 'message': 'Adjust foot position', 'tips': ['Feet hip-width apart', 'Toes slightly out', 'Weight on mid-foot'], 'color': 'warning'},
        'unknown': {'status': 'UNCLEAR POSITION', 'message': 'Position yourself in frame', 'tips': ['Ensure full body visible', 'Stand in good lighting'], 'color': 'secondary'}
    },
    'bicep_curl': {
        'none': {'status': 'EXCELLENT FORM ✓', 'message': 'Perfect bicep curl form!', 'tips': ['Elbows stable', 'Controlled movement', 'No momentum', 'Great work!'], 'color': 'success'},
        'back_too_backward_lean': {'status': 'BACKWARD LEAN ⚠', 'message': 'Do not lean backward!', 'tips': ['Engage core', 'Stand upright', 'No momentum', 'Control the weight'], 'color': 'danger'},
        'back_too_forward_lean': {'status': 'FORWARD LEAN ⚠', 'message': 'Do not lean forward!', 'tips': ['Keep torso upright', 'Shoulders back', 'Engage core'], 'color': 'danger'},
        'hand_position_too_close': {'status': 'HANDS TOO CLOSE ⚠', 'message': 'Widen your grip', 'tips': ['Hands shoulder-width apart', 'Natural grip width'], 'color': 'warning'},
        'hand_position_too_wide': {'status': 'HANDS TOO WIDE ⚠', 'message': 'Narrow your grip', 'tips': ['Bring hands closer', 'Shoulder-width grip'], 'color': 'warning'},
        'hand_above_near_head': {'status': 'OVER-CURLING ⚠', 'message': 'Do not curl too high', 'tips': ['Stop at shoulder level', 'Do not swing weights', 'Control the motion'], 'color': 'warning'},
        'one_hand_up_other_down': {'status': 'ASYMMETRIC ⚠', 'message': 'Keep both hands level', 'tips': ['Curl both arms together', 'Maintain symmetry', 'Equal weight on both sides'], 'color': 'warning'},
        'unknown': {'status': 'UNCLEAR POSITION', 'message': 'Position yourself in frame', 'tips': ['Ensure upper body visible', 'Stand in good lighting'], 'color': 'secondary'}
    }
}


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
    """Factory function to get the appropriate predictor"""
    mapping = {
        'squat': SquatPredictor,
        'lunge': LungePredictor,
        'pushup': PushupPredictor,
        'deadlift': DeadliftPredictor,
        'bicep_curl': BicepCurlPredictor,
    }
    cls = mapping.get(exercise_type)
    return cls() if cls else None


# One predictor per exercise, cached inside each worker process.
# This keeps the app working even if PythonAnywhere runs more than one
# worker, because any worker can load the model it needs on demand.
_predictors = {}


def _get_or_load(exercise_type):
    if not exercise_type:
        return None
    predictor = _predictors.get(exercise_type)
    if predictor is None:
        predictor = get_predictor_for_exercise(exercise_type)
        if predictor is None:
            return None
        if not predictor.load_model():
            return None
        _predictors[exercise_type] = predictor
    return predictor


def _attach_feedback(result, exercise_type):
    if result.get('success'):
        messages = FEEDBACK_MESSAGES.get(exercise_type, {})
        feedback = messages.get(result['prediction'], messages.get('unknown', {}))
        result['feedback'] = dict(feedback)
    return result


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/load_model', methods=['POST'])
def load_model():
    data = request.get_json(silent=True) or {}
    exercise_type = data.get('exercise_type')

    predictor = _get_or_load(exercise_type)
    if predictor is None:
        return jsonify({'success': False, 'error': f'Could not load model for "{exercise_type}"'})

    predictor.reset_counter()
    return jsonify({
        'success': True,
        'message': f'{exercise_type.title()} model ready',
        'exercise': exercise_type
    })


@app.route('/api/analyze_frame', methods=['POST'])
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

    try:
        result = predictor.analyze_landmarks(landmarks, count_rep=True)
        return jsonify(_attach_feedback(result, exercise_type))
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/analyze_image', methods=['POST'])
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

    try:
        result = predictor.analyze_image_landmarks(landmarks)
        return jsonify(_attach_feedback(result, exercise_type))
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/reset_counter', methods=['POST'])
def reset_counter():
    data = request.get_json(silent=True) or {}
    exercise_type = data.get('exercise_type')
    predictor = _predictors.get(exercise_type) if exercise_type else None

    if predictor is None:
        return jsonify({'success': True, 'message': 'Nothing to reset', 'rep_count': 0})

    predictor.reset_counter()
    return jsonify({'success': True, 'message': 'Rep counter reset', 'rep_count': 0})


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'loaded_exercises': list(_predictors.keys())
    })


if __name__ == '__main__':
    # Local testing only. PythonAnywhere uses the WSGI file, not this block.
    print("Starting Flask server. Visit: http://localhost:5000")
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)