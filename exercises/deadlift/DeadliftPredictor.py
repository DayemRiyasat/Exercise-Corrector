# exercises/deadlift/DeadliftPredictor.py 

from base_predictor import ExercisePredictor
from exercises.deadlift.RepCounter import DeadliftRepCounter
import numpy as np


class DeadliftPredictor(ExercisePredictor):
    """Deadlift exercise predictor with form correction"""
    
    def __init__(self):
        super().__init__('deadlift')
        self.rep_counter = DeadliftRepCounter(form_threshold=0.10)
    
    def extract_features(self, landmarks):
        """Extract deadlift-specific features - 152 features total"""
        features = {}
        
        # KEY POINTS
        nose = [landmarks[0].x, landmarks[0].y]
        left_shoulder = [landmarks[11].x, landmarks[11].y]
        right_shoulder = [landmarks[12].x, landmarks[12].y]
        left_elbow = [landmarks[13].x, landmarks[13].y]
        right_elbow = [landmarks[14].x, landmarks[14].y]
        left_wrist = [landmarks[15].x, landmarks[15].y]
        right_wrist = [landmarks[16].x, landmarks[16].y]
        left_hip = [landmarks[23].x, landmarks[23].y]
        right_hip = [landmarks[24].x, landmarks[24].y]
        left_knee = [landmarks[25].x, landmarks[25].y]
        right_knee = [landmarks[26].x, landmarks[26].y]
        left_ankle = [landmarks[27].x, landmarks[27].y]
        right_ankle = [landmarks[28].x, landmarks[28].y]
        
        # Midpoints
        mid_shoulder = [(left_shoulder[0] + right_shoulder[0])/2, 
                        (left_shoulder[1] + right_shoulder[1])/2]
        mid_hip = [(left_hip[0] + right_hip[0])/2, 
                   (left_hip[1] + right_hip[1])/2]
        mid_wrist = [(left_wrist[0] + right_wrist[0])/2,
                     (left_wrist[1] + right_wrist[1])/2]
        mid_ankle = [(left_ankle[0] + right_ankle[0])/2,
                     (left_ankle[1] + right_ankle[1])/2]
        mid_knee = [(left_knee[0] + right_knee[0])/2,
                    (left_knee[1] + right_knee[1])/2]
        
        # BACK & SPINE FEATURES
        features['spine_angle'] = self.calculate_angle(mid_hip, mid_shoulder, nose)
        features['back_angle'] = self.calculate_angle(mid_ankle, mid_hip, mid_shoulder)
        features['upper_back_angle'] = self.calculate_angle(nose, mid_shoulder, mid_hip)
        features['lower_back_angle'] = self.calculate_angle(mid_shoulder, mid_hip, mid_ankle)
        features['spine_straightness'] = self.calculate_distance(nose, mid_hip)
        features['torso_vertical_alignment'] = abs(mid_shoulder[0] - mid_hip[0])
        features['back_horizontal_deviation'] = mid_shoulder[0] - mid_hip[0]
        features['hip_hinge_angle'] = self.calculate_angle(mid_shoulder, mid_hip, mid_knee)
        features['shoulder_hip_distance'] = self.calculate_distance(mid_shoulder, mid_hip)
        features['hip_ankle_distance'] = self.calculate_distance(mid_hip, mid_ankle)
        features['torso_lean_forward'] = mid_shoulder[0] - mid_hip[0]
        features['torso_lean_backward'] = mid_hip[0] - mid_shoulder[0]
        features['left_hip_angle'] = self.calculate_angle(left_shoulder, left_hip, left_knee)
        features['right_hip_angle'] = self.calculate_angle(right_shoulder, right_hip, right_knee)
        features['avg_hip_angle'] = (features['left_hip_angle'] + features['right_hip_angle']) / 2
        features['hip_angle_difference'] = abs(features['left_hip_angle'] - features['right_hip_angle'])
        
        # HAND GRIP FEATURES
        features['hand_grip_width'] = self.calculate_distance(left_wrist, right_wrist)
        features['hand_grip_horizontal'] = abs(left_wrist[0] - right_wrist[0])
        features['hand_grip_vertical'] = abs(left_wrist[1] - right_wrist[1])
        shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
        features['grip_to_shoulder_ratio'] = features['hand_grip_width'] / (shoulder_width + 0.001)
        features['grip_width_normalized'] = features['hand_grip_horizontal'] / (shoulder_width + 0.001)
        features['left_hand_position_x'] = left_wrist[0]
        features['right_hand_position_x'] = right_wrist[0]
        features['left_hand_position_y'] = left_wrist[1]
        features['right_hand_position_y'] = right_wrist[1]
        features['hands_center_position'] = mid_wrist[0]
        features['left_elbow_wrist_distance'] = self.calculate_distance(left_elbow, left_wrist)
        features['right_elbow_wrist_distance'] = self.calculate_distance(right_elbow, right_wrist)
        features['left_arm_angle'] = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
        features['right_arm_angle'] = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
        features['avg_arm_angle'] = (features['left_arm_angle'] + features['right_arm_angle']) / 2
        features['left_shoulder_wrist_distance'] = self.calculate_distance(left_shoulder, left_wrist)
        features['right_shoulder_wrist_distance'] = self.calculate_distance(right_shoulder, right_wrist)
        features['hand_height_difference'] = abs(left_wrist[1] - right_wrist[1])
        features['hand_level_balance'] = abs(left_wrist[1] - right_wrist[1]) / (shoulder_width + 0.001)
        
        # LEG STANCE FEATURES
        features['leg_stance_width'] = self.calculate_distance(left_ankle, right_ankle)
        features['leg_stance_horizontal'] = abs(left_ankle[0] - right_ankle[0])
        features['leg_stance_vertical'] = abs(left_ankle[1] - right_ankle[1])
        hip_width = abs(left_hip[0] - right_hip[0])
        features['stance_to_hip_ratio'] = features['leg_stance_horizontal'] / (hip_width + 0.001)
        features['stance_width_normalized'] = features['leg_stance_horizontal'] / (shoulder_width + 0.001)
        features['left_foot_position_x'] = left_ankle[0]
        features['right_foot_position_x'] = right_ankle[0]
        features['feet_center_position'] = mid_ankle[0]
        features['left_knee_angle'] = self.calculate_angle(left_hip, left_knee, left_ankle)
        features['right_knee_angle'] = self.calculate_angle(right_hip, right_knee, right_ankle)
        features['avg_knee_angle'] = (features['left_knee_angle'] + features['right_knee_angle']) / 2
        features['knee_angle_difference'] = abs(features['left_knee_angle'] - features['right_knee_angle'])
        features['knee_separation'] = abs(left_knee[0] - right_knee[0])
        features['knee_height_difference'] = abs(left_knee[1] - right_knee[1])
        features['left_knee_forward'] = left_knee[0] - left_hip[0]
        features['right_knee_forward'] = right_knee[0] - right_hip[0]
        features['left_leg_extension'] = self.calculate_distance(left_hip, left_ankle)
        features['right_leg_extension'] = self.calculate_distance(right_hip, right_ankle)
        features['leg_extension_ratio'] = features['left_leg_extension'] / (features['right_leg_extension'] + 0.001)
        features['left_leg_straightness'] = self.calculate_angle(left_ankle, left_knee, left_hip)
        features['right_leg_straightness'] = self.calculate_angle(right_ankle, right_knee, right_hip)
        
        # BODY STABILITY
        features['hip_level_balance'] = abs(left_hip[1] - right_hip[1])
        features['shoulder_level_balance'] = abs(left_shoulder[1] - right_shoulder[1])
        features['body_tilt'] = abs(mid_shoulder[0] - mid_hip[0])
        features['ankle_level_balance'] = abs(left_ankle[1] - right_ankle[1])
        features['body_center_alignment'] = abs(mid_shoulder[0] - mid_ankle[0])
        features['weight_distribution'] = mid_hip[0] - mid_ankle[0]
        
        # BAR POSITION INDICATORS
        features['hands_over_feet'] = abs(mid_wrist[0] - mid_ankle[0])
        features['hips_over_feet'] = abs(mid_hip[0] - mid_ankle[0])
        features['shoulders_over_feet'] = abs(mid_shoulder[0] - mid_ankle[0])
        features['bar_path_vertical'] = abs(mid_wrist[1] - mid_ankle[1])
        features['bar_path_horizontal'] = abs(mid_wrist[0] - mid_ankle[0])
        
        # ADDITIONAL FEATURES
        features['shoulders_behind_bar'] = mid_shoulder[0] - mid_wrist[0]
        features['shoulders_forward_of_bar'] = mid_wrist[0] - mid_shoulder[0]
        features['hip_drop'] = mid_hip[1] - mid_shoulder[1]
        features['hip_rise'] = mid_shoulder[1] - mid_hip[1]
        features['forward_lean'] = mid_shoulder[0] - mid_hip[0]
        features['backward_lean'] = mid_hip[0] - mid_shoulder[0]
        
        # RAW LANDMARKS (13 key points × 4 = 52 features)
        key_landmarks = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
        for i in key_landmarks:
            landmark = landmarks[i]
            features[f'landmark_{i}_x'] = landmark.x
            features[f'landmark_{i}_y'] = landmark.y
            features[f'landmark_{i}_z'] = landmark.z
            features[f'landmark_{i}_visibility'] = landmark.visibility
        
        return features
    
    def get_analysis_details(self, landmarks):
        """Get detailed analysis for deadlift form"""
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_wrist = landmarks[15]
        right_wrist = landmarks[16]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        left_knee = landmarks[25]
        right_knee = landmarks[26]
        left_ankle = landmarks[27]
        right_ankle = landmarks[28]
        
        # Hip hinge angle (most important for deadlift)
        left_hip_angle = self.rep_counter.calculate_angle(left_shoulder, left_hip, left_knee)
        right_hip_angle = self.rep_counter.calculate_angle(right_shoulder, right_hip, right_knee)
        avg_hip_angle = (left_hip_angle + right_hip_angle) / 2
        
        # Back angle
        mid_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
        mid_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        mid_hip_x = (left_hip.x + right_hip.x) / 2
        mid_hip_y = (left_hip.y + right_hip.y) / 2
        mid_ankle_x = (left_ankle.x + right_ankle.x) / 2
        mid_ankle_y = (left_ankle.y + right_ankle.y) / 2
        
        back_angle = self.calculate_angle(
            [mid_ankle_x, mid_ankle_y],
            [mid_hip_x, mid_hip_y],
            [mid_shoulder_x, mid_shoulder_y]
        )
        
        # Grip width
        hand_width = abs(left_wrist.x - right_wrist.x)
        shoulder_width = abs(left_shoulder.x - right_shoulder.x)
        grip_ratio = hand_width / (shoulder_width + 0.001)
        
        # Stance width
        stance_width = abs(left_ankle.x - right_ankle.x)
        hip_width = abs(left_hip.x - right_hip.x)
        stance_ratio = stance_width / (hip_width + 0.001)
        
        # Back straightness
        back_lean = mid_shoulder_x - mid_hip_x
        
        return {
            'hip_angle': round(avg_hip_angle, 1),
            'back_angle': round(back_angle, 1),
            'grip_width_ratio': round(grip_ratio, 2),
            'grip_position': 'Good' if 0.9 <= grip_ratio <= 1.3 else ('Too narrow' if grip_ratio < 0.9 else 'Too wide'),
            'stance_width_ratio': round(stance_ratio, 2),
            'stance_position': 'Good' if 0.8 <= stance_ratio <= 1.2 else ('Too narrow' if stance_ratio < 0.8 else 'Too wide'),
            'back_straightness': 'Good' if abs(back_lean) < 0.1 else ('Forward lean' if back_lean > 0 else 'Backward lean'),
            'hip_hinge_depth': bool(avg_hip_angle <= 90)
        }
