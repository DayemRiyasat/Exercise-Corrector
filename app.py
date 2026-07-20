# app.py - Refactored Main Application
# Location: project_root/app.py

from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import os
import gc

# Import common classes
from video_processor import VideoProcessor

# Import exercise modules
from exercises.squat.SquatPredictor import SquatPredictor
from exercises.lunge.LungePredictor import LungePredictor 
from exercises.pushup.PushupPredictor import PushupPredictor
from exercises.deadlift.DeadliftPredictor import DeadliftPredictor
from exercises.bicep_curl.BicepCurlPredictor import BicepCurlPredictor

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables
current_predictor = None
video_processor = None

# Feedback messages configuration
FEEDBACK_MESSAGES = {
    'squat': {
        'none': {
            'status': 'EXCELLENT FORM ✓',
            'message': 'Perfect squat form!',
            'tips': ['Keep back straight', 'Chest up', 'Knees aligned with toes', 'Great work!'],
            'color': 'success'
        },
        'extreme_backward_lean': {
            'status': 'BACK ISSUE ⚠',
            'message': 'Don\'t lean backward',
            'tips': ['Engage core muscles', 'Maintain neutral spine', 'Keep weight centered'],
            'color': 'danger'
        },
        'extreme_forward_lean': {
            'status': 'FORWARD LEAN ⚠',
            'message': 'Don\'t lean too far forward',
            'tips': ['Keep chest up', 'Look forward', 'Sit back into the squat'],
            'color': 'warning'
        },
        'foots_too_close': {
            'status': 'STANCE TOO NARROW ⚠',
            'message': 'Widen your stance',
            'tips': ['Feet shoulder-width apart', 'Toes slightly pointed out'],
            'color': 'warning'
        },
        'foots_too_far': {
            'status': 'STANCE TOO WIDE ⚠',
            'message': 'Narrow your stance',
            'tips': ['Bring feet closer', 'Maintain control'],
            'color': 'warning'
        },
        'unknown': {
            'status': 'UNCLEAR POSITION',
            'message': 'Position yourself in frame',
            'tips': ['Ensure full body visible', 'Stand in good lighting'],
            'color': 'secondary'
        }
    },
    'lunge': {   
        'none': {
            'status': 'EXCELLENT FORM ✓',
            'message': 'Perfect lunge form!',
            'tips': ['Keep back straight', 'Front knee over ankle', 'Back knee bent', 'Great work!'],
            'color': 'success'
        },
        'extreme_backward_lean': {
            'status': 'BACK ISSUE ⚠',
            'message': 'Don\'t lean backward',
            'tips': ['Engage core muscles', 'Keep torso upright', 'Look forward'],
            'color': 'danger'
        },
        'extreme_forward_lean': {
            'status': 'FORWARD LEAN ⚠',
            'message': 'Don\'t lean too far forward',
            'tips': ['Keep chest up', 'Shoulders back', 'Stay vertical'],
            'color': 'warning'
        },
        'foots_too_close': {
            'status': 'STANCE TOO NARROW ⚠',
            'message': 'Step further forward',
            'tips': ['Increase stride length', 'Front foot should be forward'],
            'color': 'warning'
        },
        'foots_too_far': {
            'status': 'STANCE TOO WIDE ⚠',
            'message': 'Reduce stride length',
            'tips': ['Step closer', 'Maintain balance'],
            'color': 'warning'
        },
        'unknown': {
            'status': 'UNCLEAR POSITION',
            'message': 'Position yourself in frame',
            'tips': ['Ensure full body visible', 'Stand in good lighting'],
            'color': 'secondary'
        }
    },  
    'pushup': {  
        'none': {
            'status': 'EXCELLENT FORM ✓',
            'message': 'Perfect push-up form!',
            'tips': ['Elbows close to body', 'Straight back', 'Full range of motion', 'Great work!'],
            'color': 'success'
        },
        'hand_too_far_or_incorrect_position': {
            'status': 'HAND POSITION ⚠',
            'message': 'Adjust hand placement',
            'tips': ['Hands shoulder-width apart', 'Position under shoulders', 'Fingers forward'],
            'color': 'warning'
        },
        'hips_too_high': {
            'status': 'HIP POSITION ⚠',
            'message': 'Lower your hips',
            'tips': ['Maintain plank position', 'Keep core engaged', 'Straight line head to heels'],
            'color': 'warning'
        },
        'incorrect_leg_position': {
            'status': 'LEG ALIGNMENT ⚠',
            'message': 'Check leg position',
            'tips': ['Keep legs straight', 'Feet together', 'Toes on ground'],
            'color': 'warning'
        },
        'unknown': {
            'status': 'UNCLEAR POSITION',
            'message': 'Position yourself in frame',
            'tips': ['Ensure full body visible', 'Stand in good lighting'],
            'color': 'secondary'
        }
    },
     'deadlift': {   
        'none': {
            'status': 'EXCELLENT FORM ✓',
            'message': 'Perfect deadlift form!',
            'tips': ['Neutral spine', 'Chest up', 'Hips and shoulders rise together', 'Great work!'],
            'color': 'success'
        },
        'back_arch_posture': {
            'status': 'BACK ARCH - CRITICAL ⚠',
            'message': 'Keep spine neutral!',
            'tips': ['Engage core', 'Chest up', 'Don\'t hyperextend back', 'Maintain neutral spine throughout'],
            'color': 'danger'
        },
        'hand_grip_width': {
            'status': 'GRIP WIDTH ⚠',
            'message': 'Adjust hand position',
            'tips': ['Hands shoulder-width or slightly wider', 'Arms straight', 'Grip outside knees'],
            'color': 'warning'
        },
        'leg_position_width': {
            'status': 'STANCE WIDTH ⚠',
            'message': 'Adjust foot position',
            'tips': ['Feet hip-width apart', 'Toes slightly out', 'Weight on mid-foot'],
            'color': 'warning'
        },
        'unknown': {
            'status': 'UNCLEAR POSITION',
            'message': 'Position yourself in frame',
            'tips': ['Ensure full body visible', 'Stand in good lighting'],
            'color': 'secondary'
        }
    },
    'bicep_curl': {   
        'none': {
            'status': 'EXCELLENT FORM ✓',
            'message': 'Perfect bicep curl form!',
            'tips': ['Elbows stable', 'Controlled movement', 'No momentum', 'Great work!'],
            'color': 'success'
        },
        'back_too_backward_lean': {
            'status': 'BACKWARD LEAN ⚠',
            'message': 'Don\'t lean backward!',
            'tips': ['Engage core', 'Stand upright', 'No momentum', 'Control the weight'],
            'color': 'danger'
        },
        'back_too_forward_lean': {
            'status': 'FORWARD LEAN ⚠',
            'message': 'Don\'t lean forward!',
            'tips': ['Keep torso upright', 'Shoulders back', 'Engage core'],
            'color': 'danger'
        },
        'hand_position_too_close': {
            'status': 'HANDS TOO CLOSE ⚠',
            'message': 'Widen your grip',
            'tips': ['Hands shoulder-width apart', 'Natural grip width'],
            'color': 'warning'
        },
        'hand_position_too_wide': {
            'status': 'HANDS TOO WIDE ⚠',
            'message': 'Narrow your grip',
            'tips': ['Bring hands closer', 'Shoulder-width grip'],
            'color': 'warning'
        },
        'hand_above_near_head': {
            'status': 'OVER-CURLING ⚠',
            'message': 'Don\'t curl too high',
            'tips': ['Stop at shoulder level', 'Don\'t swing weights', 'Control the motion'],
            'color': 'warning'
        },
        'one_hand_up_other_down': {
            'status': 'ASYMMETRIC ⚠',
            'message': 'Keep both hands level',
            'tips': ['Curl both arms together', 'Maintain symmetry', 'Equal weight on both sides'],
            'color': 'warning'
        },
        'unknown': {
            'status': 'UNCLEAR POSITION',
            'message': 'Position yourself in frame',
            'tips': ['Ensure upper body visible', 'Stand in good lighting'],
            'color': 'secondary'
        }
    }

}

# In get_predictor_for_exercise():
def get_predictor_for_exercise(exercise_type):
    """Factory function to get the appropriate predictor"""
    if exercise_type == 'squat':
        return SquatPredictor()
    elif exercise_type == 'lunge':
        return LungePredictor()
    elif exercise_type == 'pushup':
        return PushupPredictor()
    elif exercise_type == 'deadlift':
        return DeadliftPredictor()
    elif exercise_type == 'bicep_curl':  # NEW
        return BicepCurlPredictor()
    else:
        return None
    

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/load_model', methods=['POST'])
def load_model():
    global current_predictor, video_processor
    
    data = request.get_json()
    exercise_type = data.get('exercise_type')
    
    if not exercise_type:
        return jsonify({'success': False, 'error': 'No exercise type provided'})
    
    # Cleanup existing predictor
    if current_predictor:
        current_predictor.cleanup()
        current_predictor = None
        gc.collect()
    
    if video_processor:
        video_processor.stop()
        video_processor = None
    
    # Get new predictor
    current_predictor = get_predictor_for_exercise(exercise_type)
    
    if not current_predictor:
        return jsonify({'success': False, 'error': f'Exercise type "{exercise_type}" not yet implemented'})
    
    success = current_predictor.load_model()
    
    if success:
        return jsonify({
            'success': True,
            'message': f'{exercise_type.title()} model loaded successfully',
            'exercise': exercise_type
        })
    else:
        current_predictor = None
        return jsonify({
            'success': False,
            'error': f'Failed to load {exercise_type} model'
        })


@app.route('/api/process_frame', methods=['POST'])
def process_frame():
    global current_predictor
    
    if not current_predictor:
        return jsonify({'success': False, 'error': 'No model loaded'})
    
    data = request.get_json()
    frame_data = data.get('frame')
    
    if not frame_data:
        return jsonify({'success': False, 'error': 'No frame data provided'})
    
    try:
        img_data = base64.b64decode(frame_data.split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'success': False, 'error': 'Failed to decode frame'})
        
        height, width = frame.shape[:2]
        if width > 1280:
            scale = 1280 / width
            frame = cv2.resize(frame, (1280, int(height * scale)))
        
        result = current_predictor.process_frame(frame)
        
        if result['success']:
            feedback = FEEDBACK_MESSAGES[current_predictor.exercise_type].get(
                result['prediction'],
                FEEDBACK_MESSAGES[current_predictor.exercise_type]['unknown']
            ).copy()
            result['feedback'] = feedback
        
        return jsonify(result)
            
    except Exception as e:
        print(f"Error processing frame: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/process_image', methods=['POST'])
def process_image():
    global current_predictor
    
    if not current_predictor:
        return jsonify({'success': False, 'error': 'No model loaded'})
    
    data = request.get_json()
    image_data = data.get('image')
    
    if not image_data:
        return jsonify({'success': False, 'error': 'No image data provided'})
    
    try:
        img_data = base64.b64decode(image_data.split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'success': False, 'error': 'Failed to decode image'})
        
        # Resize to standard size for better consistency
        height, width = frame.shape[:2]
        if width > 1280:
            scale = 1280 / width
            frame = cv2.resize(frame, (1280, int(height * scale)))
        elif width < 640:  # ← NEW: Upscale small images
            scale = 640 / width
            frame = cv2.resize(frame, (640, int(height * scale)))
        
        # Optional: Improve contrast (helps with poor lighting)
        # Uncomment if images have low contrast:
        # frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=10)
        
        result = current_predictor.process_image(frame)
        
        if result['success']:
            feedback = FEEDBACK_MESSAGES[current_predictor.exercise_type].get(
                result['prediction'],
                FEEDBACK_MESSAGES[current_predictor.exercise_type]['unknown']
            ).copy()
            result['feedback'] = feedback
        
        return jsonify(result)
            
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/start_video_processing', methods=['POST'])
def start_video_processing():
    global current_predictor, video_processor
    
    if not current_predictor:
        return jsonify({'success': False, 'error': 'No model loaded'})
    
    if video_processor:
        video_processor.stop()
    
    video_processor = VideoProcessor(current_predictor)
    video_processor.start()
    
    current_predictor.reset_counter()
    
    return jsonify({
        'success': True,
        'message': 'Video processor started'
    })


@app.route('/api/process_video_frame', methods=['POST'])
def process_video_frame():
    global video_processor
    
    if not video_processor:
        return jsonify({'success': False, 'error': 'Video processor not initialized'})
    
    data = request.get_json()
    frame_data = data.get('frame')
    frame_number = data.get('frame_number', 0)
    timestamp = data.get('timestamp', 0)
    
    if not frame_data:
        return jsonify({'success': False, 'error': 'No frame data provided'})
    
    success = video_processor.add_frame(frame_data, frame_number, timestamp)
    
    return jsonify({
        'success': success,
        'message': 'Frame queued' if success else 'Queue full'
    })


@app.route('/api/get_video_result', methods=['GET'])
def get_video_result():
    global video_processor
    
    if not video_processor:
        return jsonify({'success': False, 'error': 'Video processor not initialized'})
    
    result = video_processor.get_result(timeout=0.05)
    
    if result:
        if result.get('success') and result.get('prediction'):
            feedback = FEEDBACK_MESSAGES[current_predictor.exercise_type].get(
                result['prediction'],
                FEEDBACK_MESSAGES[current_predictor.exercise_type]['unknown']
            )
            result['feedback'] = feedback
        
        return jsonify(result)
    else:
        return jsonify({'success': False, 'no_result': True})


@app.route('/api/stop_video_processing', methods=['POST'])
def stop_video_processing():
    global video_processor
    
    if video_processor:
        video_processor.stop()
        video_processor = None
    
    return jsonify({
        'success': True,
        'message': 'Video processor stopped'
    })


@app.route('/api/pause_video_processing', methods=['POST'])
def pause_video_processing():
    global video_processor
    
    if not video_processor:
        return jsonify({'success': False, 'error': 'Video processor not initialized'})
    
    video_processor.pause()
    
    return jsonify({
        'success': True,
        'message': 'Video processor paused'
    })


@app.route('/api/resume_video_processing', methods=['POST'])
def resume_video_processing():
    global video_processor
    
    if not video_processor:
        return jsonify({'success': False, 'error': 'Video processor not initialized'})
    
    video_processor.resume()
    
    return jsonify({
        'success': True,
        'message': 'Video processor resumed'
    })


@app.route('/api/reset_counter', methods=['POST'])
def reset_counter():
    global current_predictor
    
    if not current_predictor:
        return jsonify({'success': False, 'error': 'No model loaded'})
    
    current_predictor.reset_counter()
    
    return jsonify({
        'success': True,
        'message': 'Rep counter reset',
        'rep_count': 0
    })


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': current_predictor is not None,
        'exercise_type': current_predictor.exercise_type if current_predictor else None,
        'rep_count': current_predictor.get_rep_count() if current_predictor else 0,
        'video_processor_active': video_processor is not None
    })


if __name__ == '__main__':
    print("=" * 60)
    print("🏋️ EXERCISE FORM CORRECTION SYSTEM")
    print("=" * 60)
    print("✨ Features:")
    print("  - Smart Rep Counter (landmark-based)")
    print("  - 10% Incorrect Form Tolerance")
    print("  - Threaded Video Processing")
    print("  - Real-time State Tracking")
    print("  - Image Upload Analysis")
    print("  - Modular Exercise Structure")
    print("=" * 60)
    print("📋 Available Exercises:")
    print("  ✅ Squat")
    print("  ✅ Lunge (NEW)")
    print("  ⏳ Push-up (Coming Soon)")
    print("  ⏳ Plank (Coming Soon)")
    print("=" * 60)
    print("Starting Flask server...")
    print("Visit: http://localhost:5000")
    print("=" * 60)
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)