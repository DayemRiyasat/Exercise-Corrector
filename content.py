# content.py - Site content and the exercise knowledge base
# Location: project_root/content.py
#
# This module is the single source of truth for everything the site says
# about an exercise: the marketing copy, the coaching library, and the
# feedback the API returns while a set is running.
#
# The model's class names are the spine of it. Every fault below maps to
# one label the trained classifier can emit, so the library on the site
# and the coaching in the app can never drift apart.

# ---------------------------------------------------------------------------
# Exercise library
# ---------------------------------------------------------------------------
# Each fault entry carries two audiences:
#   status / message / tips / color -> shown live, mid-set, in the app
#   label / why / fix               -> shown in the library, read at rest
#
# 'none' is the clean-form class. 'unknown' is the model saying it cannot
# see enough of you to judge.

EXERCISES = {
    'squat': {
        'name': 'Squat',
        'icon': 'bi-person-standing',
        'tagline': 'Back and leg analysis',
        'difficulty': 'Beginner',
        'pattern': 'Knee-dominant',
        'muscles': ['Quadriceps', 'Glutes', 'Adductors', 'Spinal erectors'],
        'equipment': 'Bodyweight or barbell',
        'summary': (
            'The squat is the clearest read on how your hips, knees and spine share load. '
            'Most squat faults are not strength problems, they are position problems, and '
            'position is exactly what a camera can see.'
        ),
        'camera': 'Film from the side at hip height, roughly three metres back, with your whole body in frame.',
        'setup': [
            'Feet about shoulder-width, toes turned out slightly.',
            'Weight spread across the whole foot, not the toes.',
            'Ribs down, core braced as if about to be nudged.',
            'Eyes on a fixed point ahead, neck neutral.',
        ],
        'execution': [
            'Break at the hips and knees together, not one then the other.',
            'Track your knees over your toes as you descend.',
            'Descend until your hip crease passes below your knee, if your mobility allows.',
            'Drive the floor away, keeping your chest angle unchanged out of the hole.',
        ],
        'faults': {
            'none': {
                'status': 'EXCELLENT FORM ✓',
                'message': 'Perfect squat form!',
                'tips': ['Keep back straight', 'Chest up', 'Knees aligned with toes', 'Great work!'],
                'color': 'success',
                'label': 'Clean rep',
                'why': 'Hips, knees and spine are sharing the load the way they should.',
                'fix': 'Nothing to change. Hold this position and add load or reps.',
            },
            'extreme_backward_lean': {
                'status': 'BACK ISSUE ⚠',
                'message': 'Do not lean backward',
                'tips': ['Engage core muscles', 'Maintain neutral spine', 'Keep weight centred'],
                'color': 'danger',
                'label': 'Leaning backward',
                'why': 'Leaning back shifts load off the hips and onto the lower spine, and it usually means the core let go at the bottom.',
                'fix': 'Brace before you descend and keep your ribs stacked over your pelvis the whole way up.',
            },
            'extreme_forward_lean': {
                'status': 'FORWARD LEAN ⚠',
                'message': 'Do not lean too far forward',
                'tips': ['Keep chest up', 'Look forward', 'Sit back into the squat'],
                'color': 'warning',
                'label': 'Leaning forward',
                'why': 'A heavy forward tip turns the squat into a good morning and loads the lower back instead of the legs.',
                'fix': 'Sit back rather than down, and check your ankle mobility. Heel-elevated squats often clear this up.',
            },
            'foots_too_close': {
                'status': 'STANCE TOO NARROW ⚠',
                'message': 'Widen your stance',
                'tips': ['Feet shoulder-width apart', 'Toes slightly pointed out'],
                'color': 'warning',
                'label': 'Stance too narrow',
                'why': 'A narrow base leaves the hips no room to travel, so depth gets stolen from the lower back.',
                'fix': 'Step out to roughly shoulder width and turn the toes out about fifteen degrees.',
            },
            'foots_too_far': {
                'status': 'STANCE TOO WIDE ⚠',
                'message': 'Narrow your stance',
                'tips': ['Bring feet closer', 'Maintain control'],
                'color': 'warning',
                'label': 'Stance too wide',
                'why': 'Too wide and the knees are pushed into the adductors, which limits depth and stresses the groin.',
                'fix': 'Bring the feet in until you can reach depth without the knees collapsing inward.',
            },
            'unknown': {
                'status': 'UNCLEAR POSITION',
                'message': 'Position yourself in frame',
                'tips': ['Ensure full body visible', 'Stand in good lighting'],
                'color': 'secondary',
                'label': 'Unclear position',
                'why': 'The model cannot see enough joints to make a call.',
                'fix': 'Step back until your head and feet are both in frame, and light yourself from the front.',
            },
        },
    },

    'lunge': {
        'name': 'Lunge',
        'icon': 'bi-person-walking',
        'tagline': 'Leg and balance analysis',
        'difficulty': 'Beginner',
        'pattern': 'Single-leg',
        'muscles': ['Quadriceps', 'Glutes', 'Hamstrings', 'Calves'],
        'equipment': 'Bodyweight or dumbbells',
        'summary': (
            'Lunges expose the side-to-side differences a squat hides. Because you are balancing '
            'on one leg, small trunk faults show up immediately, which makes them ideal for camera analysis.'
        ),
        'camera': 'Film from the side so both the front and rear leg stay visible through the whole step.',
        'setup': [
            'Stand tall with feet hip-width apart.',
            'Take a step long enough that both knees can reach ninety degrees.',
            'Keep your torso upright, shoulders over your hips.',
            'Front foot flat, rear heel lifted.',
        ],
        'execution': [
            'Lower straight down rather than pitching forward.',
            'Stop when the rear knee is just short of the floor.',
            'Keep the front shin close to vertical.',
            'Push through the front heel to return.',
        ],
        'faults': {
            'none': {
                'status': 'EXCELLENT FORM ✓',
                'message': 'Perfect lunge form!',
                'tips': ['Keep back straight', 'Front knee over ankle', 'Back knee bent', 'Great work!'],
                'color': 'success',
                'label': 'Clean rep',
                'why': 'Torso is upright and the stride length is letting both knees work.',
                'fix': 'Nothing to change. Add load or slow the descent down.',
            },
            'extreme_backward_lean': {
                'status': 'BACK ISSUE ⚠',
                'message': 'Do not lean backward',
                'tips': ['Engage core muscles', 'Keep torso upright', 'Look forward'],
                'color': 'danger',
                'label': 'Leaning backward',
                'why': 'Leaning back on a lunge unloads the front leg and puts a shear force through the lower spine.',
                'fix': 'Brace the core and think about keeping your sternum directly over your belt buckle.',
            },
            'extreme_forward_lean': {
                'status': 'FORWARD LEAN ⚠',
                'message': 'Do not lean too far forward',
                'tips': ['Keep chest up', 'Shoulders back', 'Stay vertical'],
                'color': 'warning',
                'label': 'Leaning forward',
                'why': 'Tipping forward turns a lunge into a hinge and takes the quads out of the movement.',
                'fix': 'Shorten the step slightly and lower straight down instead of reaching forward.',
            },
            'foots_too_close': {
                'status': 'STANCE TOO NARROW ⚠',
                'message': 'Step further forward',
                'tips': ['Increase stride length', 'Front foot should be forward'],
                'color': 'warning',
                'label': 'Stride too short',
                'why': 'A short stride forces the front knee well past the toes and jams the ankle.',
                'fix': 'Step a little further out so the front shin can stay near vertical at the bottom.',
            },
            'foots_too_far': {
                'status': 'STANCE TOO WIDE ⚠',
                'message': 'Reduce stride length',
                'tips': ['Step closer', 'Maintain balance'],
                'color': 'warning',
                'label': 'Stride too long',
                'why': 'An over-long stride stops the rear knee bending and pulls the hip flexors into a stretch under load.',
                'fix': 'Bring the step in until both knees can fold to about ninety degrees.',
            },
            'unknown': {
                'status': 'UNCLEAR POSITION',
                'message': 'Position yourself in frame',
                'tips': ['Ensure full body visible', 'Stand in good lighting'],
                'color': 'secondary',
                'label': 'Unclear position',
                'why': 'The model cannot see enough joints to make a call.',
                'fix': 'Step back so both legs stay in frame for the whole step.',
            },
        },
    },

    'pushup': {
        'name': 'Push-up',
        'icon': 'bi-person-arms-up',
        'tagline': 'Arm and core analysis',
        'difficulty': 'Beginner',
        'pattern': 'Horizontal push',
        'muscles': ['Pectorals', 'Triceps', 'Anterior deltoids', 'Core'],
        'equipment': 'Bodyweight',
        'summary': (
            'A push-up is a moving plank. Most of what goes wrong happens at the hips rather than '
            'the arms, and hip height is one of the easiest things to measure from a side-on camera.'
        ),
        'camera': 'Film from the side at floor level so the line from your shoulders to your heels is visible.',
        'setup': [
            'Hands just outside shoulder width, fingers pointing forward.',
            'Shoulders stacked directly over your wrists.',
            'Feet together or hip-width, whichever is steadier.',
            'Squeeze the glutes so the hips cannot sag.',
        ],
        'execution': [
            'Lower under control until your chest is near the floor.',
            'Keep the elbows at roughly forty-five degrees, not flared to ninety.',
            'Hold one straight line from head to heels the whole time.',
            'Press back up without letting the hips lead.',
        ],
        'faults': {
            'none': {
                'status': 'EXCELLENT FORM ✓',
                'message': 'Perfect push-up form!',
                'tips': ['Elbows close to body', 'Straight back', 'Full range of motion', 'Great work!'],
                'color': 'success',
                'label': 'Clean rep',
                'why': 'The body is holding one line and the arms are doing the work.',
                'fix': 'Nothing to change. Slow the tempo or elevate the feet to progress.',
            },
            'hand_too_far_or_incorrect_position': {
                'status': 'HAND POSITION ⚠',
                'message': 'Adjust hand placement',
                'tips': ['Hands shoulder-width apart', 'Position under shoulders', 'Fingers forward'],
                'color': 'warning',
                'label': 'Hand placement off',
                'why': 'Hands too far forward or too wide put the shoulder into an exposed position at the bottom.',
                'fix': 'Move the hands back under the shoulders and keep them just outside shoulder width.',
            },
            'hips_too_high': {
                'status': 'HIP POSITION ⚠',
                'message': 'Lower your hips',
                'tips': ['Maintain plank position', 'Keep core engaged', 'Straight line head to heels'],
                'color': 'warning',
                'label': 'Hips riding high',
                'why': 'Piking the hips shortens the range and lets the core opt out of the rep.',
                'fix': 'Squeeze the glutes and tuck the ribs down so the hips drop into line with the shoulders.',
            },
            'incorrect_leg_position': {
                'status': 'LEG ALIGNMENT ⚠',
                'message': 'Check leg position',
                'tips': ['Keep legs straight', 'Feet together', 'Toes on ground'],
                'color': 'warning',
                'label': 'Leg alignment off',
                'why': 'Bent or splayed legs break the straight line the push-up depends on for core tension.',
                'fix': 'Straighten the knees and bring the feet closer together to narrow the base.',
            },
            'unknown': {
                'status': 'UNCLEAR POSITION',
                'message': 'Position yourself in frame',
                'tips': ['Ensure full body visible', 'Stand in good lighting'],
                'color': 'secondary',
                'label': 'Unclear position',
                'why': 'The model cannot see enough joints to make a call.',
                'fix': 'Lower the camera towards floor level and make sure your feet are still in shot.',
            },
        },
    },

    'deadlift': {
        'name': 'Deadlift',
        'icon': 'bi-arrow-bar-up',
        'tagline': 'Back and hip analysis',
        'difficulty': 'Intermediate',
        'pattern': 'Hip-dominant',
        'muscles': ['Hamstrings', 'Glutes', 'Spinal erectors', 'Lats'],
        'equipment': 'Barbell or dumbbells',
        'summary': (
            'The deadlift carries the highest cost for a bad rep, which is why spine position is '
            'the one thing worth watching above all else. The model treats back arch as a critical fault, not a warning.'
        ),
        'camera': 'Film side-on at hip height so the angle of your spine is clearly visible off the floor.',
        'setup': [
            'Bar over mid-foot, shins close but not touching.',
            'Feet hip-width, toes slightly out.',
            'Grip just outside the knees, arms straight.',
            'Set the lats by pulling the bar into your legs before you lift.',
        ],
        'execution': [
            'Take the slack out of the bar before anything moves.',
            'Push the floor away rather than pulling with the back.',
            'Hips and shoulders rise at the same rate.',
            'Finish standing tall without leaning back at the top.',
        ],
        'faults': {
            'none': {
                'status': 'EXCELLENT FORM ✓',
                'message': 'Perfect deadlift form!',
                'tips': ['Neutral spine', 'Chest up', 'Hips and shoulders rise together', 'Great work!'],
                'color': 'success',
                'label': 'Clean rep',
                'why': 'Spine stayed neutral and the hips and shoulders moved as one unit.',
                'fix': 'Nothing to change. Add load in small steps and keep the bar path tight.',
            },
            'back_arch_posture': {
                'status': 'BACK ARCH - CRITICAL ⚠',
                'message': 'Keep spine neutral!',
                'tips': ['Engage core', 'Chest up', 'Do not hyperextend back', 'Maintain neutral spine throughout'],
                'color': 'danger',
                'label': 'Back arching',
                'why': 'Losing the neutral spine under load is the single highest-risk fault the model tracks. It concentrates force on the discs instead of the muscles.',
                'fix': 'Drop the weight. Rebuild the brace, set the lats, and only add load once the spine angle holds for every rep.',
            },
            'hand_grip_width': {
                'status': 'GRIP WIDTH ⚠',
                'message': 'Adjust hand position',
                'tips': ['Hands shoulder-width or slightly wider', 'Arms straight', 'Grip outside knees'],
                'color': 'warning',
                'label': 'Grip width off',
                'why': 'A grip that is too wide or too narrow changes the bar path and makes the lats harder to engage.',
                'fix': 'Set the hands just outside the knees so the arms hang vertically when you are in position.',
            },
            'leg_position_width': {
                'status': 'STANCE WIDTH ⚠',
                'message': 'Adjust foot position',
                'tips': ['Feet hip-width apart', 'Toes slightly out', 'Weight on mid-foot'],
                'color': 'warning',
                'label': 'Stance width off',
                'why': 'The wrong stance width pushes the hips too high or too low at the start, which changes which muscles take the first pull.',
                'fix': 'Reset to hip-width. Your shins should be close to vertical with the bar over mid-foot.',
            },
            'unknown': {
                'status': 'UNCLEAR POSITION',
                'message': 'Position yourself in frame',
                'tips': ['Ensure full body visible', 'Stand in good lighting'],
                'color': 'secondary',
                'label': 'Unclear position',
                'why': 'The model cannot see enough joints to make a call.',
                'fix': 'Move the camera side-on and clear anything between you and the lens.',
            },
        },
    },

    'bicep_curl': {
        'name': 'Bicep curl',
        'icon': 'bi-arrow-repeat',
        'tagline': 'Arm and back analysis',
        'difficulty': 'Beginner',
        'pattern': 'Elbow flexion',
        'muscles': ['Biceps brachii', 'Brachialis', 'Forearm flexors'],
        'equipment': 'Dumbbells or barbell',
        'summary': (
            'Curls look simple, which is why they are the most commonly cheated lift in the gym. '
            'The model watches the torso as much as the arms, because that is where the cheating happens.'
        ),
        'camera': 'Film from the front so both arms are visible and the model can compare them to each other.',
        'setup': [
            'Feet hip-width, knees soft.',
            'Elbows tucked against your ribs.',
            'Shoulders back and down, chest open.',
            'Grip shoulder-width with wrists neutral.',
        ],
        'execution': [
            'Curl by bending the elbow only, keeping the upper arm still.',
            'Stop at shoulder level rather than swinging past it.',
            'Lower under control for longer than you lifted.',
            'Both arms move together and arrive together.',
        ],
        'faults': {
            'none': {
                'status': 'EXCELLENT FORM ✓',
                'message': 'Perfect bicep curl form!',
                'tips': ['Elbows stable', 'Controlled movement', 'No momentum', 'Great work!'],
                'color': 'success',
                'label': 'Clean rep',
                'why': 'The torso stayed still and the elbows did all the work.',
                'fix': 'Nothing to change. Slow the lowering phase to make the same weight harder.',
            },
            'back_too_backward_lean': {
                'status': 'BACKWARD LEAN ⚠',
                'message': 'Do not lean backward!',
                'tips': ['Engage core', 'Stand upright', 'No momentum', 'Control the weight'],
                'color': 'danger',
                'label': 'Leaning backward',
                'why': 'Leaning back to start the curl swaps bicep tension for a lower-back load. It is the classic sign the weight is too heavy.',
                'fix': 'Drop the weight a notch and stand with your back against a wall to feel what still looks like.',
            },
            'back_too_forward_lean': {
                'status': 'FORWARD LEAN ⚠',
                'message': 'Do not lean forward!',
                'tips': ['Keep torso upright', 'Shoulders back', 'Engage core'],
                'color': 'danger',
                'label': 'Leaning forward',
                'why': 'Tipping forward lets the shoulders take over and shortens the range the biceps actually work through.',
                'fix': 'Stack the ribs over the hips and brace before each rep.',
            },
            'hand_position_too_close': {
                'status': 'HANDS TOO CLOSE ⚠',
                'message': 'Widen your grip',
                'tips': ['Hands shoulder-width apart', 'Natural grip width'],
                'color': 'warning',
                'label': 'Hands too close',
                'why': 'A narrow grip rolls the wrists inward and shifts work off the biceps.',
                'fix': 'Widen out to shoulder width and keep the wrists in line with the forearms.',
            },
            'hand_position_too_wide': {
                'status': 'HANDS TOO WIDE ⚠',
                'message': 'Narrow your grip',
                'tips': ['Bring hands closer', 'Shoulder-width grip'],
                'color': 'warning',
                'label': 'Hands too wide',
                'why': 'A very wide grip puts the wrists and elbows at an angle they cannot hold under load.',
                'fix': 'Bring the hands back to shoulder width so the forearms stay vertical.',
            },
            'hand_above_near_head': {
                'status': 'OVER-CURLING ⚠',
                'message': 'Do not curl too high',
                'tips': ['Stop at shoulder level', 'Do not swing weights', 'Control the motion'],
                'color': 'warning',
                'label': 'Curling too high',
                'why': 'Past shoulder height the elbow travels forward and the biceps unload at the very point you think you are peaking.',
                'fix': 'Finish the rep at shoulder level and hold for a beat before lowering.',
            },
            'one_hand_up_other_down': {
                'status': 'ASYMMETRIC ⚠',
                'message': 'Keep both hands level',
                'tips': ['Curl both arms together', 'Maintain symmetry', 'Equal weight on both sides'],
                'color': 'warning',
                'label': 'Arms out of sync',
                'why': 'One arm leading usually means a real strength difference, and training around it makes the gap wider.',
                'fix': 'Switch to single-arm curls for a few weeks and match the weaker side.',
            },
            'unknown': {
                'status': 'UNCLEAR POSITION',
                'message': 'Position yourself in frame',
                'tips': ['Ensure upper body visible', 'Stand in good lighting'],
                'color': 'secondary',
                'label': 'Unclear position',
                'why': 'The model cannot see enough joints to make a call.',
                'fix': 'Face the camera and make sure both arms stay in frame.',
            },
        },
    },
}

# Order the library and the pickers are displayed in.
EXERCISE_ORDER = ['squat', 'lunge', 'pushup', 'deadlift', 'bicep_curl']

# Classes that are not faults: one is a clean rep, one is a non-reading.
NON_FAULT_KEYS = {'none', 'unknown'}


# ---------------------------------------------------------------------------
# Derived views of the library
# ---------------------------------------------------------------------------
def feedback_messages():
    """The live-coaching payload the API attaches to each prediction.

    Rebuilt from the library so the app and the site can never disagree.
    """
    out = {}
    for slug, ex in EXERCISES.items():
        out[slug] = {
            key: {
                'status': f['status'],
                'message': f['message'],
                'tips': list(f['tips']),
                'color': f['color'],
            }
            for key, f in ex['faults'].items()
        }
    return out


def get_exercise(slug):
    """One exercise, with its slug attached. None if the slug is unknown."""
    ex = EXERCISES.get(slug)
    if not ex:
        return None
    return {**ex, 'slug': slug, 'fault_count': count_faults(slug)}


def all_exercises():
    """The library in display order."""
    return [get_exercise(slug) for slug in EXERCISE_ORDER if slug in EXERCISES]


def count_faults(slug):
    """How many real faults one exercise's model can name."""
    ex = EXERCISES.get(slug, {})
    return len([k for k in ex.get('faults', {}) if k not in NON_FAULT_KEYS])


def total_faults():
    """Every distinct fault the system can detect, across all exercises."""
    keys = set()
    for slug in EXERCISES:
        keys.update(k for k in EXERCISES[slug]['faults'] if k not in NON_FAULT_KEYS)
    return len(keys)


def fault_label(exercise_slug, fault_key):
    """Human-readable name for a model class."""
    ex = EXERCISES.get(exercise_slug, {})
    fault = ex.get('faults', {}).get(fault_key)
    if fault:
        return fault['label']
    return str(fault_key or '').replace('_', ' ').capitalize()


def site_stats():
    """Headline numbers for the landing page, computed rather than typed
    so they stay true when the library changes."""
    return {
        'exercises': len(EXERCISES),
        'landmarks': 33,           # MediaPipe pose topology
        'checks_per_second': 7,    # ~150 ms analysis interval
        'faults': total_faults(),
    }


# ---------------------------------------------------------------------------
# Marketing content
# ---------------------------------------------------------------------------
HOW_IT_WORKS = [
    {
        'icon': 'bi-hand-index-thumb',
        'title': 'Pick your lift',
        'body': 'Choose one of five exercises. Each one loads its own trained model, tuned to the faults that matter for that movement.',
    },
    {
        'icon': 'bi-camera-video',
        'title': 'Frame yourself',
        'body': 'Use your webcam, upload a clip, or check a single photo. Pose detection runs in your browser, so the video never leaves your device.',
    },
    {
        'icon': 'bi-activity',
        'title': 'Move',
        'body': 'Reps are counted automatically and every frame is classified while you lift. Corrections appear in under a second, not after the set.',
    },
    {
        'icon': 'bi-graph-up-arrow',
        'title': 'Review',
        'body': 'Each set is saved with its rep count, clean-rep percentage and the faults you hit, so you can see whether the fix is holding.',
    },
]

FEATURES = [
    {
        'icon': 'bi-stopwatch',
        'art': 'timing',
        'title': 'Correction while it still counts',
        'body': 'Feedback lands mid-set, not in a report afterwards. You get to fix the fifth rep instead of reading about it later.',
    },
    {
        'icon': 'bi-shield-lock',
        'art': 'privacy',
        'title': 'Video never leaves your device',
        'body': 'Pose detection runs in the browser. Only 33 anonymous joint coordinates are sent for classification, never the footage.',
    },
    {
        'icon': 'bi-123',
        'art': 'counting',
        'title': 'Reps counted for you',
        'body': 'A state machine per exercise tracks the movement phase, so the count follows real range of motion rather than a timer.',
    },
    {
        'icon': 'bi-clipboard-data',
        'art': 'history',
        'title': 'Every session kept',
        'body': 'Sets are stored with duration, accuracy and a fault breakdown, so progress is something you can actually check.',
    },
    {
        'icon': 'bi-book',
        'art': 'library',
        'title': 'A reason behind every flag',
        'body': 'The library explains what each fault costs you and how to fix it, rather than just telling you that something was wrong.',
    },
    {
        'icon': 'bi-laptop',
        'art': 'browser',
        'title': 'Nothing to install',
        'body': 'It runs in a browser tab on a laptop or a phone. No app store, no account, no equipment beyond a camera.',
    },
]

FAQ = [
    {
        'q': 'What does the system actually analyse?',
        'a': 'Your browser extracts 33 body landmarks from each frame using MediaPipe pose detection. Those coordinates are sent to a neural network trained for the specific exercise you picked, which classifies the rep as clean or names the fault it sees.',
    },
    {
        'q': 'Does my video get uploaded anywhere?',
        'a': 'No. Pose detection runs entirely in your browser. The only thing sent to the server is a list of anonymous joint coordinates, which cannot be turned back into an image of you. Your webcam feed and any clips you upload stay on your device.',
    },
    {
        'q': 'How accurate is it?',
        'a': 'Each exercise has its own model with its own confidence score, shown live while you lift. Treat it as a well-informed second pair of eyes rather than a verdict. Confidence below about sixty percent usually means the framing is poor, not that your form is bad.',
    },
    {
        'q': 'Can it replace a coach or a physiotherapist?',
        'a': 'No, and it is not meant to. It catches the common, visible faults that a coach would call out on the gym floor. It cannot assess pain, injury history, or anything it cannot see. If something hurts, stop and speak to a qualified professional.',
    },
    {
        'q': 'Why does it say my position is unclear?',
        'a': 'That is the model telling you it cannot see enough joints to judge. Step back until your whole body is in frame, light yourself from the front rather than behind, and stand against a plain background.',
    },
    {
        'q': 'Where should I put the camera?',
        'a': 'For squats, lunges, push-ups and deadlifts, film side-on. For bicep curls, face the camera so both arms are visible and can be compared. Each exercise page in the library gives the specific framing it expects.',
    },
    {
        'q': 'Do I need an account?',
        'a': 'No. Sessions are tied to an anonymous ID stored in a cookie on your browser, so your history follows you on that device without any sign-up. Clearing your cookies starts you fresh.',
    },
    {
        'q': 'What equipment do I need?',
        'a': 'A device with a camera and a modern browser. Squats, lunges and push-ups need nothing else. Deadlifts and curls assume you have a barbell or dumbbells, though the analysis works on the empty-handed movement too.',
    },
    {
        'q': 'Why are my reps not being counted?',
        'a': 'The rep counter waits for a full range of motion and enforces a minimum interval between reps, so partials and bounces are ignored on purpose. If nothing is counting at all, check that your whole body is in frame and that the confidence bar is not sitting near zero.',
    },
    {
        'q': 'Can I use it on my phone?',
        'a': 'Yes. The interface is built for small screens, though you will need somewhere to prop the phone up so that your whole body stays in frame while you move.',
    },
]

SAFETY_NOTE = (
    'This tool provides general form feedback from a camera and is not medical advice. '
    'It cannot see pain, injury or individual limitations. Stop if something hurts and '
    'speak to a qualified coach or clinician.'
)