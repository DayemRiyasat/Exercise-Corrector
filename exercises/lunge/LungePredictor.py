# exercises/lunge/LungePredictor.py
# Location: project_root/exercises/lunge/LungePredictor.py

from base_predictor import ExercisePredictor
from exercises.lunge.RepCounter import LungeRepCounter
import numpy as np


class LungePredictor(ExercisePredictor):
    """Lunge exercise predictor with form correction"""
    
    def __init__(self):
        super().__init__('lunge')
        self.rep_counter = LungeRepCounter(form_threshold=0.10)
    
    def extract_features(self, landmarks):
        """Extract lunge-specific features from landmarks (66 features total)"""
        features = {}
        
        # Get landmark positions (NO HANDS - matches training data)
        nose = [landmarks[0].x, landmarks[0].y]
        left_shoulder = [landmarks[11].x, landmarks[11].y]
        right_shoulder = [landmarks[12].x, landmarks[12].y]
        left_hip = [landmarks[23].x, landmarks[23].y]
        right_hip = [landmarks[24].x, landmarks[24].y]
        left_knee = [landmarks[25].x, landmarks[25].y]
        right_knee = [landmarks[26].x, landmarks[26].y]
        left_ankle = [landmarks[27].x, landmarks[27].y]
        right_ankle = [landmarks[28].x, landmarks[28].y]
        
        # Calculate midpoints
        mid_shoulder = [(left_shoulder[0] + right_shoulder[0])/2, 
                        (left_shoulder[1] + right_shoulder[1])/2]
        mid_hip = [(left_hip[0] + right_hip[0])/2, 
                   (left_hip[1] + right_hip[1])/2]
        
        # ========================================
        # CALCULATED FEATURES (30 features)
        # ========================================
        
        # KNEE ANGLES (4 features)
        features['left_knee_angle'] = self.calculate_angle(left_hip, left_knee, left_ankle)
        features['right_knee_angle'] = self.calculate_angle(right_hip, right_knee, right_ankle)
        features['avg_knee_angle'] = (features['left_knee_angle'] + features['right_knee_angle']) / 2
        features['knee_angle_diff'] = abs(features['left_knee_angle'] - features['right_knee_angle'])
        
        # HIP ANGLES (3 features)
        features['left_hip_angle'] = self.calculate_angle(left_shoulder, left_hip, left_knee)
        features['right_hip_angle'] = self.calculate_angle(right_shoulder, right_hip, right_knee)
        features['avg_hip_angle'] = (features['left_hip_angle'] + features['right_hip_angle']) / 2
        
        # SPINE ALIGNMENT (6 features)
        features['spine_angle'] = self.calculate_angle(mid_hip, mid_shoulder, nose)
        features['torso_vertical_alignment'] = abs(mid_shoulder[0] - mid_hip[0])
        features['forward_lean'] = mid_shoulder[0] - mid_hip[0]
        features['backward_lean'] = mid_hip[0] - mid_shoulder[0]
        features['spine_straightness'] = self.calculate_distance(nose, mid_hip)
        features['upper_spine_angle'] = self.calculate_angle(nose, mid_shoulder, mid_hip)
        
        # FOOT DISTANCE (4 features)
        features['foot_distance_horizontal'] = abs(left_ankle[0] - right_ankle[0])
        features['foot_distance_vertical'] = abs(left_ankle[1] - right_ankle[1])
        features['foot_distance_total'] = self.calculate_distance(left_ankle, right_ankle)
        hip_width = abs(left_hip[0] - right_hip[0])
        features['foot_distance_normalized'] = features['foot_distance_horizontal'] / (hip_width + 0.001)
        
        # KNEE POSITIONS (4 features)
        features['left_knee_forward'] = left_knee[0] - left_hip[0]
        features['right_knee_forward'] = right_knee[0] - right_hip[0]
        features['knee_separation'] = abs(left_knee[0] - right_knee[0])
        features['knee_height_diff'] = abs(left_knee[1] - right_knee[1])
        
        # ANKLE POSITIONS (3 features)
        features['left_ankle_position_x'] = left_ankle[0]
        features['right_ankle_position_x'] = right_ankle[0]
        features['ankle_width'] = abs(left_ankle[0] - right_ankle[0])
        
        # BODY STABILITY (3 features)
        features['hip_level'] = abs(left_hip[1] - right_hip[1])
        features['shoulder_level'] = abs(left_shoulder[1] - right_shoulder[1])
        features['body_tilt'] = abs(mid_shoulder[0] - mid_hip[0])
        
        # LEG EXTENSION (3 features)
        features['left_leg_extension'] = self.calculate_distance(left_hip, left_ankle)
        features['right_leg_extension'] = self.calculate_distance(right_hip, right_ankle)
        features['leg_extension_ratio'] = features['left_leg_extension'] / (features['right_leg_extension'] + 0.001)
        
        # ========================================
        # RAW LANDMARK FEATURES (36 features)
        # 9 landmarks × 4 values (x, y, z, visibility) = 36 features
        # ========================================
        key_landmarks = [0, 11, 12, 23, 24, 25, 26, 27, 28]
        for i in key_landmarks:
            landmark = landmarks[i]
            features[f'landmark_{i}_x'] = landmark.x
            features[f'landmark_{i}_y'] = landmark.y
            features[f'landmark_{i}_z'] = landmark.z
            features[f'landmark_{i}_visibility'] = landmark.visibility
        
        # Total: 30 calculated + 36 landmark = 66 features
        
        return features
    
    def get_analysis_details(self, landmarks):
        """Get detailed analysis for lunge form"""
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        left_knee = landmarks[25]
        right_knee = landmarks[26]
        left_ankle = landmarks[27]
        right_ankle = landmarks[28]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        
        # Detect forward leg
        forward_leg = 'left' if left_ankle.y > right_ankle.y else 'right'
        
        # Calculate angles based on forward leg
        if forward_leg == 'left':
            front_knee_angle = self.rep_counter.calculate_angle(left_hip, left_knee, left_ankle)
            back_knee_angle = self.rep_counter.calculate_angle(right_hip, right_knee, right_ankle)
            front_hip_angle = self.calculate_angle([left_shoulder.x, left_shoulder.y], 
                                                   [left_hip.x, left_hip.y], 
                                                   [left_knee.x, left_knee.y])
        else:
            front_knee_angle = self.rep_counter.calculate_angle(right_hip, right_knee, right_ankle)
            back_knee_angle = self.rep_counter.calculate_angle(left_hip, left_knee, left_ankle)
            front_hip_angle = self.calculate_angle([right_shoulder.x, right_shoulder.y], 
                                                   [right_hip.x, right_hip.y], 
                                                   [right_knee.x, right_knee.y])
        
        # Check lunge depth (front knee should bend ~90°)
        depth_achieved = front_knee_angle <= 110
        
        # Foot stance
        foot_distance_vertical = abs(left_ankle.y - right_ankle.y)
        stance_assessment = 'Good' if 0.15 <= foot_distance_vertical <= 0.35 else \
                           ('Too close' if foot_distance_vertical < 0.15 else 'Too far')
        
        # Back alignment
        mid_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
        mid_hip_x = (left_hip.x + right_hip.x) / 2
        lean = mid_shoulder_x - mid_hip_x
        
        return {
            'front_knee_angle': round(float(front_knee_angle), 1),
            'back_knee_angle': round(float(back_knee_angle), 1),
            'front_hip_angle': round(float(front_hip_angle), 1),
            'forward_leg': forward_leg,
            'depth_achieved': bool(depth_achieved),  # Convert numpy bool to Python bool
            'stance_width': stance_assessment,
            'back_lean': 'Forward' if lean > 0.05 else ('Backward' if lean < -0.05 else 'Neutral'),
            'lean_amount': abs(round(float(lean * 100), 1))
        }