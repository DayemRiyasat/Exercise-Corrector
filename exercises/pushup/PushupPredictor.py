# exercises/pushup/PushupPredictor.py

from base_predictor import ExercisePredictor
from exercises.pushup.RepCounter import PushupRepCounter
import numpy as np


class PushupPredictor(ExercisePredictor):
    """Push-up exercise predictor with form correction"""
    
    def __init__(self):
        super().__init__('pushup')
        self.rep_counter = PushupRepCounter(form_threshold=0.10)
    
    def extract_features(self, landmarks):
        """Extract push-up specific features - 122 features total"""
        features = {}
        
        # Key points
        nose = [landmarks[0].x, landmarks[0].y]
        left_wrist = [landmarks[15].x, landmarks[15].y]
        right_wrist = [landmarks[16].x, landmarks[16].y]
        left_elbow = [landmarks[13].x, landmarks[13].y]
        right_elbow = [landmarks[14].x, landmarks[14].y]
        left_shoulder = [landmarks[11].x, landmarks[11].y]
        right_shoulder = [landmarks[12].x, landmarks[12].y]
        left_hip = [landmarks[23].x, landmarks[23].y]
        right_hip = [landmarks[24].x, landmarks[24].y]
        left_knee = [landmarks[25].x, landmarks[25].y]
        right_knee = [landmarks[26].x, landmarks[26].y]
        left_ankle = [landmarks[27].x, landmarks[27].y]
        right_ankle = [landmarks[28].x, landmarks[28].y]
        left_heel = [landmarks[29].x, landmarks[29].y]
        right_heel = [landmarks[30].x, landmarks[30].y]
        
        # Midpoints
        mid_shoulder = [(left_shoulder[0] + right_shoulder[0])/2, 
                        (left_shoulder[1] + right_shoulder[1])/2]
        mid_hip = [(left_hip[0] + right_hip[0])/2, 
                   (left_hip[1] + right_hip[1])/2]
        mid_wrist = [(left_wrist[0] + right_wrist[0])/2,
                     (left_wrist[1] + right_wrist[1])/2]
        mid_ankle = [(left_ankle[0] + right_ankle[0])/2,
                     (left_ankle[1] + right_ankle[1])/2]
        
        # 1. HAND POSITION FEATURES (12 features)
        hand_width = abs(left_wrist[0] - right_wrist[0])
        shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
        features['hand_width'] = hand_width
        features['hand_shoulder_width_ratio'] = hand_width / (shoulder_width + 0.001)
        features['left_hand_forward'] = left_wrist[0] - left_shoulder[0]
        features['right_hand_forward'] = right_wrist[0] - right_shoulder[0]
        features['hand_forward_asymmetry'] = abs(features['left_hand_forward'] - features['right_hand_forward'])
        features['hand_height_diff'] = abs(left_wrist[1] - right_wrist[1])
        features['hand_vertical_position'] = mid_wrist[1] - mid_shoulder[1]
        features['left_hand_distance_from_center'] = abs(left_wrist[0] - mid_shoulder[0])
        features['right_hand_distance_from_center'] = abs(right_wrist[0] - mid_shoulder[0])
        features['left_wrist_shoulder_alignment'] = self.calculate_distance(left_wrist, left_shoulder)
        features['right_wrist_shoulder_alignment'] = self.calculate_distance(right_wrist, right_shoulder)
        features['wrist_shoulder_alignment_diff'] = abs(features['left_wrist_shoulder_alignment'] - 
                                                         features['right_wrist_shoulder_alignment'])
        
        # 2. ELBOW ANGLES (6 features)
        features['left_elbow_angle'] = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
        features['right_elbow_angle'] = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
        features['avg_elbow_angle'] = (features['left_elbow_angle'] + features['right_elbow_angle']) / 2
        features['elbow_angle_diff'] = abs(features['left_elbow_angle'] - features['right_elbow_angle'])
        features['elbow_symmetry'] = 1.0 / (1.0 + features['elbow_angle_diff'])
        features['elbow_spread'] = abs(left_elbow[0] - right_elbow[0])
        
        # 3. SHOULDER ANGLES (4 features)
        features['left_shoulder_angle'] = self.calculate_angle(left_elbow, left_shoulder, left_hip)
        features['right_shoulder_angle'] = self.calculate_angle(right_elbow, right_shoulder, right_hip)
        features['avg_shoulder_angle'] = (features['left_shoulder_angle'] + features['right_shoulder_angle']) / 2
        features['shoulder_angle_diff'] = abs(features['left_shoulder_angle'] - features['right_shoulder_angle'])
        
        # 4. BACK/SPINE ALIGNMENT (10 features)
        features['body_line_angle_upper'] = self.calculate_angle(nose, mid_shoulder, mid_hip)
        features['body_line_angle_lower'] = self.calculate_angle(mid_shoulder, mid_hip, mid_ankle)
        features['full_body_line_angle'] = self.calculate_angle(mid_shoulder, mid_hip, mid_ankle)
        features['plank_straightness'] = self.calculate_angle(mid_shoulder, mid_hip, mid_ankle)
        features['spine_deviation'] = abs(180 - features['plank_straightness'])
        features['hip_elevation_vs_shoulder'] = mid_hip[1] - mid_shoulder[1]
        features['hip_elevation_vs_ankle'] = mid_hip[1] - mid_ankle[1]
        features['hip_sag_indicator'] = mid_hip[1] - ((mid_shoulder[1] + mid_ankle[1]) / 2)
        features['back_arch'] = self.calculate_distance(mid_hip, [(mid_shoulder[0] + mid_ankle[0])/2,
                                                                   (mid_shoulder[1] + mid_ankle[1])/2])
        features['torso_alignment'] = abs(mid_shoulder[0] - mid_hip[0])
        
        # 5. HIP POSITION (8 features)
        features['hip_height_normalized'] = mid_hip[1] / (mid_shoulder[1] + 0.001)
        features['hip_shoulder_vertical_distance'] = abs(mid_hip[1] - mid_shoulder[1])
        features['hip_ankle_vertical_distance'] = abs(mid_hip[1] - mid_ankle[1])
        features['left_hip_angle'] = self.calculate_angle(left_shoulder, left_hip, left_knee)
        features['right_hip_angle'] = self.calculate_angle(right_shoulder, right_hip, right_knee)
        features['avg_hip_angle'] = (features['left_hip_angle'] + features['right_hip_angle']) / 2
        features['hip_angle_diff'] = abs(features['left_hip_angle'] - features['right_hip_angle'])
        features['hip_level_balance'] = abs(left_hip[1] - right_hip[1])
        
        # 6. LEG POSITION (10 features)
        features['left_knee_angle'] = self.calculate_angle(left_hip, left_knee, left_ankle)
        features['right_knee_angle'] = self.calculate_angle(right_hip, right_knee, right_ankle)
        features['avg_knee_angle'] = (features['left_knee_angle'] + features['right_knee_angle']) / 2
        features['knee_angle_diff'] = abs(features['left_knee_angle'] - features['right_knee_angle'])
        features['leg_straightness_score'] = (features['left_knee_angle'] + features['right_knee_angle']) / 2
        features['leg_bend_penalty'] = abs(180 - features['leg_straightness_score'])
        features['leg_width_at_knee'] = abs(left_knee[0] - right_knee[0])
        features['leg_width_at_ankle'] = abs(left_ankle[0] - right_ankle[0])
        features['leg_width_consistency'] = abs(features['leg_width_at_knee'] - features['leg_width_at_ankle'])
        features['leg_alignment_diff'] = abs(left_knee[1] - right_knee[1])
        
        # 7. ANKLE & FOOT POSITION (6 features)
        features['ankle_width'] = abs(left_ankle[0] - right_ankle[0])
        features['ankle_height_diff'] = abs(left_ankle[1] - right_ankle[1])
        features['foot_position_symmetry'] = 1.0 / (1.0 + features['ankle_height_diff'])
        features['heel_distance'] = self.calculate_distance(left_heel, right_heel)
        features['left_foot_angle'] = self.calculate_angle(left_knee, left_ankle, left_heel)
        features['right_foot_angle'] = self.calculate_angle(right_knee, right_ankle, right_heel)
        
        # 8. BODY STABILITY & BALANCE (6 features)
        features['shoulder_level'] = abs(left_shoulder[1] - right_shoulder[1])
        features['overall_body_tilt'] = abs(mid_shoulder[0] - mid_ankle[0])
        features['center_of_mass_x'] = (mid_shoulder[0] + mid_hip[0] + mid_ankle[0]) / 3
        features['center_of_mass_y'] = (mid_shoulder[1] + mid_hip[1] + mid_ankle[1]) / 3
        features['body_balance_score'] = abs(features['center_of_mass_x'] - mid_hip[0])
        features['vertical_alignment_score'] = abs(mid_shoulder[0] - mid_hip[0]) + abs(mid_hip[0] - mid_ankle[0])
        
        # 9. RAW LANDMARK FEATURES (60 features)
        key_landmarks = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30]
        
        for i in key_landmarks:
            landmark = landmarks[i]
            features[f'landmark_{i}_x'] = landmark.x
            features[f'landmark_{i}_y'] = landmark.y
            features[f'landmark_{i}_z'] = landmark.z
            features[f'landmark_{i}_visibility'] = landmark.visibility
        
        return features
    
    def get_analysis_details(self, landmarks):
        """Get detailed analysis for push-up form"""
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_elbow = landmarks[13]
        right_elbow = landmarks[14]
        left_wrist = landmarks[15]
        right_wrist = landmarks[16]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        left_ankle = landmarks[27]
        right_ankle = landmarks[28]
        
        # Elbow angles
        left_elbow_angle = self.rep_counter.calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = self.rep_counter.calculate_angle(right_shoulder, right_elbow, right_wrist)
        avg_elbow_angle = (left_elbow_angle + right_elbow_angle) / 2
        
        # Hip position
        mid_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        mid_hip_y = (left_hip.y + right_hip.y) / 2
        mid_ankle_y = (left_ankle.y + right_ankle.y) / 2
        
        hip_alignment = 'Good' if abs(mid_hip_y - (mid_shoulder_y + mid_ankle_y)/2) < 0.1 else 'Too high' if mid_hip_y < (mid_shoulder_y + mid_ankle_y)/2 else 'Sagging'
        
        # Hand position
        hand_width = abs(left_wrist.x - right_wrist.x)
        shoulder_width = abs(left_shoulder.x - right_shoulder.x)
        hand_ratio = hand_width / (shoulder_width + 0.001)
        
        return {
            'elbow_angle': round(avg_elbow_angle, 1),
            'left_elbow_angle': round(left_elbow_angle, 1),
            'right_elbow_angle': round(right_elbow_angle, 1),
            'hip_alignment': hip_alignment,
            'hand_width_ratio': round(hand_ratio, 2),
            'hand_position': 'Good' if 0.9 <= hand_ratio <= 1.3 else ('Too narrow' if hand_ratio < 0.9 else 'Too wide'),
            'depth_achieved': bool(avg_elbow_angle <= 90)  # ← FIX HERE
        }