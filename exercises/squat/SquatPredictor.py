# exercises/squat/SquatPredictor.py
# Location: project_root/exercises/squat/SquatPredictor.py

from base_predictor import ExercisePredictor
from exercises.squat.RepCounter import SquatRepCounter
import numpy as np


class SquatPredictor(ExercisePredictor):
    """Squat exercise predictor with form correction"""
    
    def __init__(self):
        super().__init__('squat')
        self.rep_counter = SquatRepCounter(form_threshold=0.10)
    
    def extract_features(self, landmarks):
        """Extract squat-specific features from landmarks"""
        features = {}
        
        # Get landmark positions
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
        mid_knee = [(left_knee[0] + right_knee[0])/2,
                    (left_knee[1] + right_knee[1])/2]
        
        # Spine and torso features
        features['spine_angle'] = self.calculate_angle(mid_hip, mid_shoulder, nose)
        features['spine_straightness'] = self.calculate_distance(nose, mid_hip)
        features['upper_spine_angle'] = self.calculate_angle(nose, mid_shoulder, mid_hip)
        features['lower_spine_angle'] = self.calculate_angle(mid_shoulder, mid_hip, mid_knee)
        
        features['forward_lean'] = mid_shoulder[0] - mid_hip[0]
        features['backward_lean'] = mid_hip[0] - mid_shoulder[0]
        features['torso_vertical_alignment'] = abs(mid_shoulder[0] - mid_hip[0])
        features['back_arch_level'] = (mid_shoulder[0] - mid_hip[0]) / (abs(mid_shoulder[1] - mid_hip[1]) + 0.001)
        features['torso_length'] = self.calculate_distance(mid_shoulder, mid_hip)
        features['torso_to_nose_distance'] = self.calculate_distance(mid_shoulder, nose)
        features['shoulder_hip_horizontal_offset'] = abs(mid_shoulder[0] - mid_hip[0])
        features['shoulder_hip_vertical_distance'] = abs(mid_shoulder[1] - mid_hip[1])
        
        shoulder_hip_dist = self.calculate_distance(mid_shoulder, mid_hip)
        vertical_component = abs(mid_shoulder[1] - mid_hip[1])
        features['back_straightness_score'] = vertical_component / (shoulder_hip_dist + 0.001)
        
        # Knee angles
        features['left_knee_angle'] = self.calculate_angle(left_hip, left_knee, left_ankle)
        features['right_knee_angle'] = self.calculate_angle(right_hip, right_knee, right_ankle)
        features['avg_knee_angle'] = (features['left_knee_angle'] + features['right_knee_angle']) / 2
        features['knee_angle_diff'] = abs(features['left_knee_angle'] - features['right_knee_angle'])
        
        # Hip angles
        features['left_hip_angle'] = self.calculate_angle(left_shoulder, left_hip, left_knee)
        features['right_hip_angle'] = self.calculate_angle(right_shoulder, right_hip, right_knee)
        features['avg_hip_angle'] = (features['left_hip_angle'] + features['right_hip_angle']) / 2
        features['hip_angle_diff'] = abs(features['left_hip_angle'] - features['right_hip_angle'])
        
        # Ankle angles
        features['left_ankle_angle'] = self.calculate_angle(left_knee, left_ankle, 
                                                            [left_ankle[0], left_ankle[1] + 0.1])
        features['right_ankle_angle'] = self.calculate_angle(right_knee, right_ankle, 
                                                             [right_ankle[0], right_ankle[1] + 0.1])
        
        # Foot positioning
        features['foot_distance_horizontal'] = abs(left_ankle[0] - right_ankle[0])
        features['foot_distance_vertical'] = abs(left_ankle[1] - right_ankle[1])
        features['foot_distance_total'] = self.calculate_distance(left_ankle, right_ankle)
        
        hip_width = abs(left_hip[0] - right_hip[0])
        features['foot_distance_normalized'] = features['foot_distance_horizontal'] / (hip_width + 0.001)
        
        # Knee over ankle
        features['left_knee_over_ankle'] = left_knee[0] - left_ankle[0]
        features['right_knee_over_ankle'] = right_knee[0] - right_ankle[0]
        features['avg_knee_over_ankle'] = (features['left_knee_over_ankle'] + features['right_knee_over_ankle']) / 2
        
        # Knee separation
        features['knee_separation'] = abs(left_knee[0] - right_knee[0])
        features['knee_height_diff'] = abs(left_knee[1] - right_knee[1])
        
        # Leg alignment
        features['left_leg_alignment'] = abs(left_hip[0] - left_knee[0]) + abs(left_knee[0] - left_ankle[0])
        features['right_leg_alignment'] = abs(right_hip[0] - right_knee[0]) + abs(right_knee[0] - right_ankle[0])
        
        # Squat depth
        features['hip_height'] = (left_hip[1] + right_hip[1]) / 2
        features['knee_height'] = (left_knee[1] + right_knee[1]) / 2
        features['squat_depth'] = features['hip_height'] - features['knee_height']
        
        # Body balance
        features['hip_level'] = abs(left_hip[1] - right_hip[1])
        features['shoulder_level'] = abs(left_shoulder[1] - right_shoulder[1])
        features['body_tilt'] = abs(mid_shoulder[0] - mid_hip[0])
        features['ankle_level'] = abs(left_ankle[1] - right_ankle[1])
        
        # Center of mass
        features['center_of_mass_x'] = (mid_shoulder[0] + mid_hip[0]) / 2
        features['center_of_mass_y'] = (mid_shoulder[1] + mid_hip[1]) / 2
        
        # Key landmark coordinates
        key_landmarks = [0, 11, 12, 23, 24, 25, 26, 27, 28]
        for i in key_landmarks:
            landmark = landmarks[i]
            features[f'landmark_{i}_x'] = landmark.x
            features[f'landmark_{i}_y'] = landmark.y
            features[f'landmark_{i}_z'] = landmark.z
            features[f'landmark_{i}_visibility'] = landmark.visibility
        
        return features
    
    def get_analysis_details(self, landmarks):
        """Get detailed analysis for squat form"""
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        left_knee = landmarks[25]
        right_knee = landmarks[26]
        left_ankle = landmarks[27]
        right_ankle = landmarks[28]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        
        # Calculate angles
        left_knee_angle = self.rep_counter.calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = self.rep_counter.calculate_angle(right_hip, right_knee, right_ankle)
        avg_knee_angle = (left_knee_angle + right_knee_angle) / 2
        
        left_hip_angle = self.calculate_angle([left_shoulder.x, left_shoulder.y], 
                                               [left_hip.x, left_hip.y], 
                                               [left_knee.x, left_knee.y])
        right_hip_angle = self.calculate_angle([right_shoulder.x, right_shoulder.y], 
                                                [right_hip.x, right_hip.y], 
                                                [right_knee.x, right_knee.y])
        
        # Check squat depth
        hip_height = (left_hip.y + right_hip.y) / 2
        knee_height = (left_knee.y + right_knee.y) / 2
        depth_achieved = hip_height >= knee_height
        
        # Foot stance
        foot_distance = abs(left_ankle.x - right_ankle.x)
        hip_width = abs(left_hip.x - right_hip.x)
        stance_ratio = foot_distance / (hip_width + 0.001)
        
        # Back alignment
        mid_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
        mid_hip_x = (left_hip.x + right_hip.x) / 2
        lean = mid_shoulder_x - mid_hip_x
        
        return {
            'knee_angle': round(avg_knee_angle, 1),
            'left_knee_angle': round(left_knee_angle, 1),
            'right_knee_angle': round(right_knee_angle, 1),
            'hip_angle': round((left_hip_angle + right_hip_angle) / 2, 1),
            'depth_achieved': depth_achieved,
            'stance_width': 'Good' if 0.8 <= stance_ratio <= 1.5 else ('Too narrow' if stance_ratio < 0.8 else 'Too wide'),
            'back_lean': 'Forward' if lean > 0.05 else ('Backward' if lean < -0.05 else 'Neutral'),
            'lean_amount': abs(round(lean * 100, 1))
        }