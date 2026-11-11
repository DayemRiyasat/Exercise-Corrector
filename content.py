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
#
# Every exercise is marked 'engine': 'v5', which means it is judged by the
# Exercise-Corrector pipeline in cv_v5/. Squat, lunge and push-up are judged
# per whole rep. Exercises marked 'unit': 'window' (jump rope, plank) have no
# reps and are judged every 2 seconds instead. The fault keys are the
# pipeline's fault list, plus 'form_issue' for a rep or window it is sure is
# not clean without one fault standing out. 'photo': False means one photo
# cannot show the exercise, so the analyser does not offer it.

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
        'engine': 'v5',
        'camera': (
            'Film from the front or at a slight angle, phone at hip height, about three metres back, '
            'with your whole body in frame. Front-on shows stance and knees caving in best; '
            'a side angle shows lean and knees past the toes.'
        ),
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
                'message': 'Clean squat!',
                'tips': ['Keep the chest up', 'Knees in line with toes', 'Reach the same depth every rep', 'Great work!'],
                'color': 'success',
                'label': 'Clean rep',
                'why': 'Depth, stance, knee path and torso angle all stayed inside what a clean rep looks like.',
                'fix': 'Nothing to change. Hold this position and add load or reps.',
            },
            'shallow': {
                'status': 'GO DEEPER ⚠',
                'message': 'Go deeper',
                'tips': ['Sit down until your thighs reach parallel', 'Keep your heels down at the bottom'],
                'color': 'warning',
                'label': 'Too shallow',
                'why': 'Stopping short of parallel takes the glutes and adductors out of the lift and leaves most of the work on the knees.',
                'fix': 'Lower until your hip crease is level with your knee. If you cannot get there, widen the stance a little or elevate the heels.',
            },
            'knees_inward': {
                'status': 'KNEES CAVING IN ⚠',
                'message': 'Knees are caving in, push them out',
                'tips': ['Push your knees out over your toes', 'Spread the floor with your feet'],
                'color': 'danger',
                'label': 'Knees caving in',
                'why': 'Knees collapsing inward twist the knee joint under load and usually mean the hips are not doing their share.',
                'fix': 'Think about spreading the floor apart with your feet and keep the knees tracking over the middle toes the whole way.',
            },
            'torso_lean': {
                'status': 'FORWARD LEAN ⚠',
                'message': 'Torso leans too far forward, keep the chest up',
                'tips': ['Keep the chest up', 'Brace before you descend', 'Sit between your heels'],
                'color': 'warning',
                'label': 'Leaning forward',
                'why': 'A heavy forward tip turns the squat into a good morning and loads the lower back instead of the legs.',
                'fix': 'Brace first, keep the chest proud and check your ankle mobility. Heel-elevated squats often clear this up.',
            },
            'knees_over_toes': {
                'status': 'KNEES TOO FAR FORWARD ⚠',
                'message': 'Knees travel too far past the toes, sit back',
                'tips': ['Start the squat by pushing the hips back', 'Keep the weight over mid-foot'],
                'color': 'warning',
                'label': 'Knees too far forward',
                'why': 'When the knees shoot far past the toes the weight drifts onto the forefoot and the knees take load the hips should carry.',
                'fix': 'Break at the hips and knees together and keep your weight over the middle of the foot.',
            },
            'stance_wide': {
                'status': 'STANCE TOO WIDE ⚠',
                'message': 'Feet are too wide, bring them to shoulder width',
                'tips': ['Bring your feet to shoulder width', 'Toes turned out slightly'],
                'color': 'warning',
                'label': 'Stance too wide',
                'why': 'Too wide and the knees are pushed into the adductors, which limits depth and stresses the groin.',
                'fix': 'Bring the feet in to about shoulder width, toes turned out around fifteen degrees.',
            },
            'stance_narrow': {
                'status': 'STANCE TOO NARROW ⚠',
                'message': 'Feet are too close, widen to shoulder width',
                'tips': ['Feet shoulder-width apart', 'Toes slightly pointed out'],
                'color': 'warning',
                'label': 'Stance too narrow',
                'why': 'A narrow base leaves the hips no room to travel, so depth gets stolen from the lower back.',
                'fix': 'Step out to roughly shoulder width and turn the toes out about fifteen degrees.',
            },
            'form_issue': {
                'status': 'FORM CHECK ⚠',
                'message': 'This rep does not look like a clean squat',
                'tips': ['Slow down and control the descent', 'Check depth, stance and chest position'],
                'color': 'warning',
                'label': 'Form differs from a clean rep',
                'why': 'The model is confident the rep was not clean but no single fault stood out.',
                'fix': 'Film a few slow reps and compare them with the setup and execution points above.',
            },
            'unknown': {
                'status': 'UNCLEAR POSITION',
                'message': 'Position yourself in frame',
                'tips': ['Ensure full body visible', 'Stand in good lighting'],
                'color': 'secondary',
                'label': 'Unclear position',
                'why': 'The pose model cannot see enough of you to make a call.',
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
        'engine': 'v5',
        'camera': (
            'Film from the front or at a slight angle so both legs stay visible through the whole step. '
            'A side angle shows the front knee and the back knee best.'
        ),
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
                'message': 'Clean lunge!',
                'tips': ['Torso upright', 'Front knee over ankle', 'Back knee bent', 'Great work!'],
                'color': 'success',
                'label': 'Clean rep',
                'why': 'Depth, torso angle and both knees stayed inside what a clean rep looks like.',
                'fix': 'Nothing to change. Add load or slow the descent down.',
            },
            'shallow': {
                'status': 'GO DEEPER ⚠',
                'message': 'Go deeper',
                'tips': ['Lower until the back knee nearly touches the floor', 'Both knees near ninety degrees'],
                'color': 'warning',
                'label': 'Too shallow',
                'why': 'A short lunge never loads the front glute properly and turns the exercise into a bounce.',
                'fix': 'Lower straight down until the back knee is just short of the floor.',
            },
            'knee_over_toes': {
                'status': 'FRONT KNEE TOO FAR ⚠',
                'message': 'Front knee travels past the toes, shorten or sit back',
                'tips': ['Keep the front shin close to vertical', 'Take a slightly longer step'],
                'color': 'warning',
                'label': 'Front knee too far forward',
                'why': 'When the front knee drives well past the toes the knee takes the load and the heel tends to lift.',
                'fix': 'Step a little further out and lower straight down so the front shin stays near vertical.',
            },
            'torso_lean': {
                'status': 'FORWARD LEAN ⚠',
                'message': 'Torso leans too far forward, keep the chest up',
                'tips': ['Keep chest up', 'Shoulders over hips', 'Stay vertical'],
                'color': 'warning',
                'label': 'Leaning forward',
                'why': 'Tipping forward turns a lunge into a hinge and takes the quads out of the movement.',
                'fix': 'Brace the core and keep your shoulders stacked over your hips as you lower.',
            },
            'back_knee_straight': {
                'status': 'BEND THE BACK KNEE ⚠',
                'message': 'Back knee is not bending, drop it toward the floor',
                'tips': ['Drop the back knee toward the floor', 'Rear heel lifted'],
                'color': 'warning',
                'label': 'Back knee too straight',
                'why': 'A straight back leg pulls the hip flexors into a stretch under load and stops you reaching depth.',
                'fix': 'Think about lowering the back knee straight down rather than pushing the body forward.',
            },
            'form_issue': {
                'status': 'FORM CHECK ⚠',
                'message': 'This rep does not look like a clean lunge',
                'tips': ['Slow down and control the descent', 'Check depth and torso position'],
                'color': 'warning',
                'label': 'Form differs from a clean rep',
                'why': 'The model is confident the rep was not clean but no single fault stood out.',
                'fix': 'Film a few slow reps and compare them with the setup and execution points above.',
            },
            'unknown': {
                'status': 'UNCLEAR POSITION',
                'message': 'Position yourself in frame',
                'tips': ['Ensure full body visible', 'Stand in good lighting'],
                'color': 'secondary',
                'label': 'Unclear position',
                'why': 'The pose model cannot see enough of you to make a call.',
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
        'engine': 'v5',
        'camera': 'Film from the side or at a slight angle at floor level so the line from your shoulders to your heels is visible.',
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
                'message': 'Clean push-up!',
                'tips': ['Straight line head to heels', 'Full range of motion', 'Great work!'],
                'color': 'success',
                'label': 'Clean rep',
                'why': 'The body held one line and the arms worked through the full range.',
                'fix': 'Nothing to change. Slow the tempo or elevate the feet to progress.',
            },
            'shallow': {
                'status': 'GO DEEPER ⚠',
                'message': 'Go deeper',
                'tips': ['Lower until the chest is near the floor', 'Elbows to about ninety degrees'],
                'color': 'warning',
                'label': 'Too shallow',
                'why': 'Half reps skip the hardest part of the push-up, which is where the chest does most of its work.',
                'fix': 'Lower under control until your chest is a fist away from the floor, then press.',
            },
            'hip_sag': {
                'status': 'HIPS SAGGING ⚠',
                'message': 'Hips are sagging, brace the core',
                'tips': ['Squeeze the glutes', 'Brace as if about to be nudged'],
                'color': 'danger',
                'label': 'Hips sagging',
                'why': 'Sagging hips put the lower back into extension under load and mean the core has switched off.',
                'fix': 'Squeeze the glutes and brace the abs so the hips stay in line with the shoulders and heels.',
            },
            'hip_pike': {
                'status': 'HIPS TOO HIGH ⚠',
                'message': 'Hips are too high, keep a straight line',
                'tips': ['Maintain plank position', 'Straight line head to heels'],
                'color': 'warning',
                'label': 'Hips riding high',
                'why': 'Piking the hips shortens the range and lets the core opt out of the rep.',
                'fix': 'Tuck the ribs down and let the hips drop into line with the shoulders.',
            },
            'head_down': {
                'status': 'HEAD DROPPING ⚠',
                'message': 'Head is dropping, keep the neck neutral',
                'tips': ['Look at the floor just ahead of your hands', 'Chest leads, not the chin'],
                'color': 'warning',
                'label': 'Head dropping',
                'why': 'Reaching for the floor with the head fakes depth and strains the neck.',
                'fix': 'Keep the head in line with the spine and let the chest, not the face, approach the floor.',
            },
            'no_lockout': {
                'status': 'NO LOCKOUT ⚠',
                'message': 'Arms do not fully extend at the top',
                'tips': ['Press all the way up', 'Finish each rep with straight arms'],
                'color': 'warning',
                'label': 'No lockout',
                'why': 'Stopping short at the top cuts the range and keeps the triceps from finishing the rep.',
                'fix': 'Press until the elbows are straight before starting the next rep.',
            },
            'form_issue': {
                'status': 'FORM CHECK ⚠',
                'message': 'This rep does not look like a clean push-up',
                'tips': ['Slow down', 'Hold one straight line head to heels'],
                'color': 'warning',
                'label': 'Form differs from a clean rep',
                'why': 'The model is confident the rep was not clean but no single fault stood out.',
                'fix': 'Film a few slow reps and compare them with the setup and execution points above.',
            },
            'unknown': {
                'status': 'UNCLEAR POSITION',
                'message': 'Position yourself in frame',
                'tips': ['Ensure full body visible', 'Stand in good lighting'],
                'color': 'secondary',
                'label': 'Unclear position',
                'why': 'The pose model cannot see enough of you to make a call.',
                'fix': 'Lower the camera towards floor level and make sure your feet are still in shot.',
            },
        },
    },

    'jump_rope': {
        'name': 'Jump Rope',
        'icon': 'bi-arrow-repeat',
        'tagline': 'Footwork and rhythm analysis',
        'difficulty': 'Beginner',
        'pattern': 'Conditioning',
        'muscles': ['Calves', 'Quadriceps', 'Forearms', 'Shoulders'],
        'equipment': 'Skipping rope, or none',
        'summary': (
            'Good skipping is small, quiet and regular. Jumps are too quick to judge one at a time, so the '
            'model looks at two seconds of movement at once: how your feet leave the floor, whether you '
            'stay on one spot, and whether the wrists are turning.'
        ),
        'engine': 'v5',
        'unit': 'window',
        'photo': False,
        'camera': (
            'Face the camera with the phone at waist height, far enough back that your whole body '
            'including your feet stays in frame at the top of every jump.'
        ),
        'setup': [
            'Stand tall with feet hip-width or closer.',
            'Elbows close to your ribs, hands just in front of your hips.',
            'Weight on the balls of the feet, knees soft.',
            'Eyes forward, shoulders relaxed and down.',
        ],
        'execution': [
            'Turn the rope with small circles from the wrists, not big circles from the shoulders.',
            'Jump just high enough to clear the rope.',
            'Land softly on the balls of the feet, heels staying just off the floor.',
            'Stay on one spot and keep the rhythm even.',
        ],
        'faults': {
            'none': {
                'status': 'EXCELLENT FORM ✓',
                'message': 'Clean skipping!',
                'tips': ['Small, quiet jumps', 'Land on the balls of the feet', 'Keep the wrists turning', 'Great work!'],
                'color': 'success',
                'label': 'Clean form',
                'why': 'The feet left the floor, the jumps stayed on one spot and the wrists kept turning.',
                'fix': 'Nothing to change. Hold the rhythm for longer or pick up the pace.',
            },
            'no_jump': {
                'status': 'NOT JUMPING ⚠',
                'message': 'Feet are not leaving the floor, hop a little higher',
                'tips': ['Hop just high enough to clear the rope', 'Spring from the ankles'],
                'color': 'warning',
                'label': 'Feet not leaving the floor',
                'why': 'Without a real hop the calves and ankles do not load and land, which is where most of the training effect comes from.',
                'fix': 'Bounce on the spot from the ankles until both feet clear the floor by a couple of centimetres on every turn.',
            },
            'heels_only': {
                'status': 'HEELS ONLY ⚠',
                'message': 'Only the heels are moving, push off from the balls of the feet',
                'tips': ['Push off from the balls of the feet', 'Keep the heels just off the floor'],
                'color': 'warning',
                'label': 'Only the heels moving',
                'why': 'Lifting only the heels looks like jumping but the toes never leave the floor, so the rope has nothing to pass under.',
                'fix': 'Shift your weight forward onto the balls of the feet and push the whole foot off the floor.',
            },
            'side_to_side': {
                'status': 'DRIFTING SIDEWAYS ⚠',
                'message': 'Jumping side to side, stay on one spot',
                'tips': ['Stay on one spot', 'Keep the hips over the feet'],
                'color': 'warning',
                'label': 'Jumping side to side',
                'why': 'Travelling from side to side wastes energy and makes every landing a little off balance.',
                'fix': 'Pick a mark on the floor and land on it every time, with your hips stacked over your feet.',
            },
            'no_wrist_circles': {
                'status': 'WRISTS NOT TURNING ⚠',
                'message': 'Turn the rope with small wrist circles',
                'tips': ['Small circles from the wrists', 'Elbows close to your ribs'],
                'color': 'warning',
                'label': 'No wrist circles',
                'why': 'When the wrists stop turning the shoulders take over, the rope slows down and the arms tire first.',
                'fix': 'Keep the elbows in and draw small circles with your hands, even when you skip without a rope.',
            },
            'form_issue': {
                'status': 'FORM CHECK ⚠',
                'message': 'This stretch does not look like clean skipping',
                'tips': ['Slow down and find an even rhythm', 'Check your feet and wrists'],
                'color': 'warning',
                'label': 'Form differs from clean skipping',
                'why': 'The model is confident the movement was not clean but no single fault stood out.',
                'fix': 'Film a short slow set and compare it with the setup and execution points above.',
            },
            'unknown': {
                'status': 'UNCLEAR POSITION',
                'message': 'Position yourself in frame',
                'tips': ['Ensure full body visible', 'Stand in good lighting'],
                'color': 'secondary',
                'label': 'Unclear position',
                'why': 'The pose model cannot see enough of you to make a call.',
                'fix': 'Step back until your head and feet stay in frame at the top of every jump.',
            },
        },
    },

    'plank': {
        'name': 'Plank',
        'icon': 'bi-stopwatch',
        'tagline': 'Core and alignment analysis',
        'difficulty': 'Beginner',
        'pattern': 'Isometric hold',
        'muscles': ['Abdominals', 'Obliques', 'Glutes', 'Shoulders'],
        'equipment': 'Bodyweight',
        'summary': (
            'The elbow plank has no reps, only a position that either holds or slowly gives way. '
            'The model checks that position every two seconds, so you see the moment your hips start '
            'to drop rather than finding out afterwards.'
        ),
        'engine': 'v5',
        'unit': 'window',
        'camera': (
            'Film front-on with the camera at floor level, about 2 to 3 metres in front of you, '
            'so your whole body is visible. Side-on filming is not supported yet.'
        ),
        'setup': [
            'Forearms on the floor, elbows directly under the shoulders.',
            'Legs straight, feet together or hip-width.',
            'Squeeze the glutes and brace the abs before you settle.',
            'Eyes on the floor just ahead of your hands.',
        ],
        'execution': [
            'Hold one straight line from your head to your heels.',
            'Push the floor away so the shoulders stay away from your ears.',
            'Keep breathing, short and steady, without letting the brace go.',
            'Stop the hold when the line goes, not when the clock says.',
        ],
        'faults': {
            'none': {
                'status': 'EXCELLENT FORM ✓',
                'message': 'Clean plank!',
                'tips': ['Straight line head to heels', 'Elbows under the shoulders', 'Keep breathing', 'Great work!'],
                'color': 'success',
                'label': 'Clean form',
                'why': 'Hips, head and shoulders all stayed inside what a clean hold looks like.',
                'fix': 'Nothing to change. Add time or lift one foot to progress.',
            },
            'hip_sag': {
                'status': 'HIPS SAGGING ⚠',
                'message': 'Hips are sagging, brace the core',
                'tips': ['Squeeze the glutes', 'Tuck the tailbone slightly'],
                'color': 'danger',
                'label': 'Hips sagging',
                'why': 'Sagging hips hand the hold over to the lower back, which is the opposite of what a plank is for.',
                'fix': 'Squeeze the glutes and brace the abs so the hips lift back into line. If they keep dropping, end the hold.',
            },
            'hip_pike': {
                'status': 'HIPS TOO HIGH ⚠',
                'message': 'Hips are too high, keep a straight line',
                'tips': ['Lower the hips into line', 'Straight line head to heels'],
                'color': 'warning',
                'label': 'Hips riding high',
                'why': 'Piking the hips moves the weight onto the shoulders and lets the abs off the hook.',
                'fix': 'Tuck the ribs down and let the hips drop into line with the shoulders and heels.',
            },
            'head_down': {
                'status': 'HEAD DROPPING ⚠',
                'message': 'Head is dropping, keep the neck neutral',
                'tips': ['Look at the floor just ahead of your hands', 'Long back of the neck'],
                'color': 'warning',
                'label': 'Head dropping',
                'why': 'Letting the head hang strains the neck and usually pulls the shoulders round with it.',
                'fix': 'Lift the head until your ears line up with your shoulders.',
            },
            'head_up': {
                'status': 'HEAD LIFTED ⚠',
                'message': 'Head is lifted, look at the floor just ahead of your hands',
                'tips': ['Look at the floor just ahead of your hands', 'Tuck the chin slightly'],
                'color': 'warning',
                'label': 'Head lifted',
                'why': 'Looking forward crunches the back of the neck for the whole length of the hold.',
                'fix': 'Tuck the chin slightly and rest your eyes on the floor just ahead of your hands.',
            },
            'on_knees': {
                'status': 'KNEES DOWN ⚠',
                'message': 'Knees are on the floor, lift them and hold a straight line',
                'tips': ['Lift the knees', 'Push back through the heels'],
                'color': 'warning',
                'label': 'Knees on the floor',
                'why': 'With the knees down the hold gets much shorter and easier, so the abs do far less work.',
                'fix': 'Straighten the legs and lift the knees. If you need the easier version, end the set there and rest.',
            },
            'shoulders_shrugged': {
                'status': 'SHOULDERS SHRUGGED ⚠',
                'message': 'Shoulders are shrugged toward the ears, push them down',
                'tips': ['Push the floor away', 'Long neck, shoulders down'],
                'color': 'warning',
                'label': 'Shoulders shrugged',
                'why': 'Shrugging sinks the chest between the shoulders and loads the neck instead of the shoulder blades.',
                'fix': 'Press the forearms into the floor and spread the shoulder blades apart.',
            },
            'arms_wide': {
                'status': 'ELBOWS TOO WIDE ⚠',
                'message': 'Elbows are too wide, place them under the shoulders',
                'tips': ['Elbows under the shoulders', 'Forearms parallel'],
                'color': 'warning',
                'label': 'Elbows too wide',
                'why': 'Elbows outside the shoulders take the arms out of the stack and put extra strain on the front of the shoulder.',
                'fix': 'Bring the elbows in until they sit directly below your shoulders.',
            },
            'form_issue': {
                'status': 'FORM CHECK ⚠',
                'message': 'This stretch does not look like a clean plank',
                'tips': ['Reset your position', 'Hold one straight line head to heels'],
                'color': 'warning',
                'label': 'Form differs from a clean hold',
                'why': 'The model is confident the hold was not clean but no single fault stood out.',
                'fix': 'Film a short hold and compare it with the setup and execution points above.',
            },
            'unknown': {
                'status': 'UNCLEAR POSITION',
                'message': 'Position yourself in frame',
                'tips': ['Ensure full body visible', 'Stay in good lighting'],
                'color': 'secondary',
                'label': 'Unclear position',
                'why': 'The pose model cannot see enough of you to make a call.',
                'fix': 'Lower the camera to floor level and make sure your feet are still in shot.',
            },
        },
    },
}

# Order the library and the pickers are displayed in.
EXERCISE_ORDER = ['squat', 'lunge', 'pushup', 'jump_rope', 'plank']

# Classes that are not faults: one is a clean rep, one is a non-reading.
NON_FAULT_KEYS = {'none', 'unknown', 'form_issue'}

# Fault keys the earlier models used, including the retired deadlift and
# bicep curl. Sessions saved before the switch still carry them, so the
# dashboard keeps a readable name for each.
LEGACY_FAULT_LABELS = {
    'extreme_backward_lean': 'Leaning backward',
    'extreme_forward_lean': 'Leaning forward',
    'foots_too_close': 'Stance too narrow',
    'foots_too_far': 'Stance too wide',
    'hand_too_far_or_incorrect_position': 'Hand placement off',
    'hips_too_high': 'Hips riding high',
    'incorrect_leg_position': 'Leg alignment off',
    'back_arch_posture': 'Back arching',
    'hand_grip_width': 'Grip width off',
    'leg_position_width': 'Stance width off',
    'back_too_backward_lean': 'Leaning backward',
    'back_too_forward_lean': 'Leaning forward',
    'hand_position_too_close': 'Hands too close',
    'hand_position_too_wide': 'Hands too wide',
    'hand_above_near_head': 'Curling too high',
    'one_hand_up_other_down': 'Arms out of sync',
}

# Exercises that are gone from the library but may be in saved sessions.
LEGACY_EXERCISES = {
    'deadlift': {'name': 'Deadlift', 'icon': 'bi-arrow-bar-up', 'unit': 'rep', 'retired': True},
    'bicep_curl': {'name': 'Bicep curl', 'icon': 'bi-arrow-repeat', 'unit': 'rep', 'retired': True},
}


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
    return {**ex, 'slug': slug, 'fault_count': count_faults(slug),
            'unit': ex.get('unit', 'rep'), 'photo': ex.get('photo', True)}


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
    if fault_key in LEGACY_FAULT_LABELS:
        return LEGACY_FAULT_LABELS[fault_key]
    return str(fault_key or '').replace('_', ' ').capitalize()


def site_stats():
    """Headline numbers for the landing page, computed rather than typed
    so they stay true when the library changes."""
    return {
        'exercises': len(EXERCISES),
        'landmarks': 17,           # YOLO11 pose (COCO) joints
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
        'body': 'Use your webcam, upload a clip, or check a single photo. Frames are analysed and thrown away, never stored.',
    },
    {
        'icon': 'bi-activity',
        'title': 'Move',
        'body': 'Reps are counted automatically and each one is judged as soon as you finish it, so corrections arrive during the set, not after it. Jump rope and plank are judged every two seconds.',
    },
    {
        'icon': 'bi-graph-up-arrow',
        'title': 'Review',
        'body': 'Each set is saved with its count, clean percentage and the faults you hit, so you can see whether the fix is holding.',
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
        'title': 'Video is never stored',
        'body': 'Frames are analysed on the server and discarded straight away. Nothing you film is kept or written to your history.',
    },
    {
        'icon': 'bi-123',
        'art': 'counting',
        'title': 'Reps counted for you',
        'body': 'The count follows real range of motion rather than a timer. Skipping counts your jumps and a plank times your hold instead.',
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
        'a': 'A YOLO11 pose model on the server finds your joints in every frame. Each finished rep is then judged by rules on your joint angles together with a neural network trained on thousands of labelled reps, which calls the rep clean or names the faults it sees. Jump rope and plank have no reps, so the same checks run on every two seconds of movement.',
    },
    {
        'q': 'Does my video get uploaded anywhere?',
        'a': 'Yes: frames, clips and photos are sent to the server so the pose model can analyse them, and they are discarded as soon as that is done. Nothing is stored.',
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
        'a': 'For squats and lunges, film from the front or at a slight angle with your whole body in frame. For push-ups, film side-on or at a slight angle at floor level. For jump rope, face the camera with the phone at waist height and your feet in frame. For planks, film front-on with the camera at floor level, about 2 to 3 metres in front of you, so your whole body is visible. Side-on filming is not supported yet for planks. Each exercise page in the library gives the specific framing it expects.',
    },
    {
        'q': 'Do I need an account?',
        'a': 'No. Sessions are tied to an anonymous ID stored in a cookie on your browser, so your history follows you on that device without any sign-up. Clearing your cookies starts you fresh.',
    },
    {
        'q': 'What equipment do I need?',
        'a': 'A device with a camera and a modern browser. Nothing else is needed. Squats and lunges can be loaded if you want, and jump rope can be done with a rope or without one.',
    },
    {
        'q': 'Why are my reps not being counted?',
        'a': 'The rep counter waits for a full range of motion and enforces a minimum interval between reps, so partials and bounces are ignored on purpose. Jump rope and plank are not counted in reps: they are judged every two seconds. If nothing is counting at all, check that your whole body is in frame and that the confidence bar is not sitting near zero.',
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