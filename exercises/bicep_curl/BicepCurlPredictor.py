# exercises/bicep_curl/BicepCurlPredictor.py 

from base_predictor import ExercisePredictor
from exercises.bicep_curl.RepCounter import BicepCurlRepCounter
import numpy as np


class BicepCurlPredictor(ExercisePredictor):
    """Bicep curl exercise predictor with form correction"""
    
    def __init__(self):
        super().__init__('bicep_curl')
        self.rep_counter = BicepCurlRepCounter(form_threshold=0.10)
    
    def extract_features(self, landmarks):
        """Extract bicep curl specific features - 106 features total"""
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
        
        # Midpoints
        mid_shoulder = [(left_shoulder[0] + right_shoulder[0])/2, 
                        (left_shoulder[1] + right_shoulder[1])/2]
        mid_hip = [(left_hip[0] + right_hip[0])/2, 
                   (left_hip[1] + right_hip[1])/2]
        mid_wrist = [(left_wrist[0] + right_wrist[0])/2,
                     (left_wrist[1] + right_wrist[1])/2]
        mid_elbow = [(left_elbow[0] + right_elbow[0])/2,
                     (left_elbow[1] + right_elbow[1])/2]
        
        # BACK & SPINE FEATURES
        features['spine_angle'] = self.calculate_angle(mid_hip, mid_shoulder, nose)
        features['back_angle'] = self.calculate_angle(mid_shoulder, mid_hip, [mid_hip[0], mid_hip[1] + 0.1])
        features['torso_angle'] = self.calculate_angle([mid_shoulder[0], mid_shoulder[1] - 0.1], mid_shoulder, mid_hip)
        features['back_horizontal_deviation'] = mid_shoulder[0] - mid_hip[0]
        features['forward_lean'] = max(0, mid_shoulder[0] - mid_hip[0])
        features['backward_lean'] = max(0, mid_hip[0] - mid_shoulder[0])
        features['torso_vertical_alignment'] = abs(mid_shoulder[0] - mid_hip[0])
        features['spine_straightness'] = self.calculate_distance(nose, mid_hip)
        features['shoulder_hip_distance'] = self.calculate_distance(mid_shoulder, mid_hip)
        features['upper_body_tilt'] = abs(mid_shoulder[0] - mid_hip[0]) / (features['shoulder_hip_distance'] + 0.001)
        features['hip_level'] = abs(left_hip[1] - right_hip[1])
        features['shoulder_level'] = abs(left_shoulder[1] - right_shoulder[1])
        
        # ARM & ELBOW FEATURES
        features['left_elbow_angle'] = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
        features['right_elbow_angle'] = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
        features['avg_elbow_angle'] = (features['left_elbow_angle'] + features['right_elbow_angle']) / 2
        features['elbow_angle_difference'] = abs(features['left_elbow_angle'] - features['right_elbow_angle'])
        features['left_elbow_position_x'] = left_elbow[0]
        features['right_elbow_position_x'] = right_elbow[0]
        features['left_elbow_position_y'] = left_elbow[1]
        features['right_elbow_position_y'] = right_elbow[1]
        features['left_elbow_shoulder_distance'] = self.calculate_distance(left_elbow, left_shoulder)
        features['right_elbow_shoulder_distance'] = self.calculate_distance(right_elbow, right_shoulder)
        features['left_elbow_forward_of_shoulder'] = left_elbow[0] - left_shoulder[0]
        features['right_elbow_forward_of_shoulder'] = right_elbow[0] - right_shoulder[0]
        features['elbow_width'] = abs(left_elbow[0] - right_elbow[0])
        features['elbow_height_difference'] = abs(left_elbow[1] - right_elbow[1])
        
        # HAND/WRIST FEATURES
        features['left_wrist_position_x'] = left_wrist[0]
        features['right_wrist_position_x'] = right_wrist[0]
        features['left_wrist_position_y'] = left_wrist[1]
        features['right_wrist_position_y'] = right_wrist[1]
        features['hand_spacing_horizontal'] = abs(left_wrist[0] - right_wrist[0])
        features['hand_spacing_vertical'] = abs(left_wrist[1] - right_wrist[1])
        features['hand_spacing_total'] = self.calculate_distance(left_wrist, right_wrist)
        shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
        features['hand_spacing_normalized'] = features['hand_spacing_horizontal'] / (shoulder_width + 0.001)
        features['elbow_width_normalized'] = features['elbow_width'] / (shoulder_width + 0.001)
        features['left_hand_height_above_shoulder'] = left_shoulder[1] - left_wrist[1]
        features['right_hand_height_above_shoulder'] = right_shoulder[1] - right_wrist[1]
        features['avg_hand_height_above_shoulder'] = (features['left_hand_height_above_shoulder'] + 
                                                       features['right_hand_height_above_shoulder']) / 2
        features['left_hand_to_nose_distance'] = self.calculate_distance(left_wrist, nose)
        features['right_hand_to_nose_distance'] = self.calculate_distance(right_wrist, nose)
        features['avg_hand_to_nose_distance'] = (features['left_hand_to_nose_distance'] + 
                                                  features['right_hand_to_nose_distance']) / 2
        features['hand_height_symmetry'] = abs(left_wrist[1] - right_wrist[1])
        features['hand_height_symmetry_ratio'] = features['hand_height_symmetry'] / (shoulder_width + 0.001)
        
        # ARM LENGTH & EXTENSION
        features['left_upper_arm_length'] = self.calculate_distance(left_shoulder, left_elbow)
        features['right_upper_arm_length'] = self.calculate_distance(right_shoulder, right_elbow)
        features['left_forearm_length'] = self.calculate_distance(left_elbow, left_wrist)
        features['right_forearm_length'] = self.calculate_distance(right_elbow, right_wrist)
        features['left_total_arm_length'] = self.calculate_distance(left_shoulder, left_wrist)
        features['right_total_arm_length'] = self.calculate_distance(right_shoulder, right_wrist)
        features['left_arm_extension_ratio'] = features['left_total_arm_length'] / (features['left_upper_arm_length'] + features['left_forearm_length'] + 0.001)
        features['right_arm_extension_ratio'] = features['right_total_arm_length'] / (features['right_upper_arm_length'] + features['right_forearm_length'] + 0.001)
        
        # SHOULDER ANGLES
        features['left_shoulder_angle'] = self.calculate_angle(mid_hip, left_shoulder, left_elbow)
        features['right_shoulder_angle'] = self.calculate_angle(mid_hip, right_shoulder, right_elbow)
        features['avg_shoulder_angle'] = (features['left_shoulder_angle'] + features['right_shoulder_angle']) / 2
        features['shoulders_forward_of_hips'] = mid_shoulder[0] - mid_hip[0]
        features['shoulders_vertical_alignment'] = abs(mid_shoulder[0] - mid_hip[0])
        
        # BODY BALANCE & STABILITY
        features['body_center_alignment'] = abs(mid_shoulder[0] - mid_hip[0])
        features['upper_body_tilt_angle'] = self.calculate_angle([mid_hip[0], mid_hip[1] - 0.1], mid_hip, mid_shoulder)
        
        # MOVEMENT RANGE INDICATORS
        features['left_wrist_to_shoulder'] = self.calculate_distance(left_wrist, left_shoulder)
        features['right_wrist_to_shoulder'] = self.calculate_distance(right_wrist, right_shoulder)
        features['left_curl_completion'] = 180 - features['left_elbow_angle']
        features['right_curl_completion'] = 180 - features['right_elbow_angle']
        features['avg_curl_completion'] = (features['left_curl_completion'] + features['right_curl_completion']) / 2
        
        # RAW LANDMARKS (UPPER BODY ONLY - 9 landmarks × 4 = 36 features)
        key_landmarks = [0, 11, 12, 13, 14, 15, 16, 23, 24]
        for i in key_landmarks:
            landmark = landmarks[i]
            features[f'landmark_{i}_x'] = landmark.x
            features[f'landmark_{i}_y'] = landmark.y
            features[f'landmark_{i}_z'] = landmark.z
            features[f'landmark_{i}_visibility'] = landmark.visibility
        
        return features
    
    def get_analysis_details(self, landmarks):
        """Get detailed analysis for bicep curl form"""
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_elbow = landmarks[13]
        right_elbow = landmarks[14]
        left_wrist = landmarks[15]
        right_wrist = landmarks[16]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        
        # Elbow angles
        left_elbow_angle = self.rep_counter.calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = self.rep_counter.calculate_angle(right_shoulder, right_elbow, right_wrist)
        avg_elbow_angle = (left_elbow_angle + right_elbow_angle) / 2
        
        # Hand spacing
        hand_width = abs(left_wrist.x - right_wrist.x)
        shoulder_width = abs(left_shoulder.x - right_shoulder.x)
        hand_ratio = hand_width / (shoulder_width + 0.001)
        
        # Back position
        mid_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
        mid_hip_x = (left_hip.x + right_hip.x) / 2
        lean = mid_shoulder_x - mid_hip_x
        
        # Hand height symmetry
        hand_height_diff = abs(left_wrist.y - right_wrist.y)
        
        return {
            'elbow_angle': round(avg_elbow_angle, 1),
            'left_elbow_angle': round(left_elbow_angle, 1),
            'right_elbow_angle': round(right_elbow_angle, 1),
            'hand_spacing_ratio': round(hand_ratio, 2),
            'hand_spacing': 'Good' if 0.8 <= hand_ratio <= 1.2 else ('Too close' if hand_ratio < 0.8 else 'Too wide'),
            'back_position': 'Good' if abs(lean) < 0.08 else ('Forward lean' if lean > 0 else 'Backward lean'),
            'lean_amount': abs(round(lean * 100, 1)),
            'arm_symmetry': 'Good' if abs(left_elbow_angle - right_elbow_angle) < 15 else 'Asymmetric',
            'hand_height_symmetry': round(hand_height_diff * 100, 1)
        }
