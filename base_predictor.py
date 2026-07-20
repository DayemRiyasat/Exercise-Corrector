# base_predictor.py - Common base class for all exercises
# Location: project_root/base_predictor.py

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import joblib
import os
import gc
import base64

# Global drawing specifications
mp_drawing_spec_landmark = mp.solutions.drawing_utils.DrawingSpec(
    color=(0, 255, 0), thickness=2, circle_radius=2
)
mp_drawing_spec_connection = mp.solutions.drawing_utils.DrawingSpec(
    color=(0, 0, 255), thickness=2
)


class ExercisePredictor:
    """Base class for all exercise predictors"""
    
    def __init__(self, exercise_type):
        self.exercise_type = exercise_type
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.mp_pose = mp.solutions.pose
        self.pose = None
        self.mp_drawing = mp.solutions.drawing_utils
        self.rep_counter = None
        
    def load_model(self):
        """Load ML model, scaler, and label encoder"""
        try:
            model_path = f"exercises/{self.exercise_type}/models/{self.exercise_type}_model.h5"
            scaler_path = f"exercises/{self.exercise_type}/models/{self.exercise_type}_scaler.pkl"
            encoder_path = f"exercises/{self.exercise_type}/models/{self.exercise_type}_label_encoder.pkl"
            
            if not all(os.path.exists(p) for p in [model_path, scaler_path, encoder_path]):
                print(f"Model files not found for {self.exercise_type}")
                return False
            
            self.model = tf.keras.models.load_model(model_path)
            self.scaler = joblib.load(scaler_path)
            self.label_encoder = joblib.load(encoder_path)
            
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            print(f"✅ Model loaded successfully: {self.exercise_type}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def calculate_angle(self, point1, point2, point3):
        """Calculate angle between three points"""
        try:
            a = np.array(point1)
            b = np.array(point2)
            c = np.array(point3)
            
            ba = a - b
            bc = c - b
            
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
            cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
            angle = np.arccos(cosine_angle)
            return np.degrees(angle)
        except:
            return 0.0
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        try:
            return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
        except:
            return 0.0
    
    def extract_features(self, landmarks):
        """Extract features from landmarks - must be implemented by subclass"""
        raise NotImplementedError("Subclass must implement extract_features method")
    
    def predict(self, landmarks):
        """Make prediction from landmarks"""
        try:
            features_dict = self.extract_features(landmarks)
            feature_array = np.array(list(features_dict.values())).reshape(1, -1)
            feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)
            
            features_scaled = self.scaler.transform(feature_array)
            prediction_proba = self.model.predict(features_scaled, verbose=0)
            predicted_class_idx = np.argmax(prediction_proba)
            confidence = float(prediction_proba[0][predicted_class_idx])
            
            predicted_class = self.label_encoder.classes_[predicted_class_idx]
            
            return predicted_class, confidence
        except Exception as e:
            print(f"Prediction error: {e}")
            return "unknown", 0.0
    
    def process_frame(self, frame):
        """Process a single frame and return results"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    mp_drawing_spec_landmark,
                    mp_drawing_spec_connection
                )
                
                prediction, confidence = self.predict(results.pose_landmarks.landmark)
                
                rep_info = self.rep_counter.update(
                    results.pose_landmarks.landmark,
                    prediction,
                    confidence
                )
                
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                processed_frame = base64.b64encode(buffer).decode('utf-8')
                
                return {
                    'success': True,
                    'processed_frame': f'data:image/jpeg;base64,{processed_frame}',
                    'prediction': prediction,
                    'confidence': confidence,
                    'rep_info': rep_info
                }
            else:
                return {
                    'success': False,
                    'error': 'No pose detected',
                    'message': 'Position yourself in frame'
                }
                
        except Exception as e:
            print(f"Error processing frame: {e}")
            return {'success': False, 'error': str(e)}
    
    def process_image(self, frame):
        """Process a single image and return detailed analysis"""
        try:
            # Create a separate Pose instance for static images
            pose_static = self.mp_pose.Pose(
                static_image_mode=True,      # ← CRITICAL: True for images
                model_complexity=2,          # ← Higher quality for single images
                smooth_landmarks=False,      # ← No smoothing for static images
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_static.process(rgb_frame)
            
            # Close the static pose instance
            pose_static.close()
            
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    mp_drawing_spec_landmark,
                    mp_drawing_spec_connection
                )
                
                prediction, confidence = self.predict(results.pose_landmarks.landmark)
                analysis_details = self.get_analysis_details(results.pose_landmarks.landmark)
                
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                processed_frame = base64.b64encode(buffer).decode('utf-8')
                
                return {
                    'success': True,
                    'processed_frame': f'data:image/jpeg;base64,{processed_frame}',
                    'prediction': prediction,
                    'confidence': confidence,
                    'analysis_details': analysis_details
                }
            else:
                return {
                    'success': False,
                    'error': 'No pose detected',
                    'message': 'Ensure full body is visible in the image'
                }
                
        except Exception as e:
            print(f"Error processing image: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
        
        
    # def process_image(self, frame):
    #     """Process a single image and return detailed analysis"""
    #     try:
    #         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         results = self.pose.process(rgb_frame)
            
    #         if results.pose_landmarks:
    #             self.mp_drawing.draw_landmarks(
    #                 frame,
    #                 results.pose_landmarks,
    #                 self.mp_pose.POSE_CONNECTIONS,
    #                 mp_drawing_spec_landmark,
    #                 mp_drawing_spec_connection
    #             )
                
    #             prediction, confidence = self.predict(results.pose_landmarks.landmark)
    #             analysis_details = self.get_analysis_details(results.pose_landmarks.landmark)
                
    #             _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    #             processed_frame = base64.b64encode(buffer).decode('utf-8')
                
    #             return {
    #                 'success': True,
    #                 'processed_frame': f'data:image/jpeg;base64,{processed_frame}',
    #                 'prediction': prediction,
    #                 'confidence': confidence,
    #                 'analysis_details': analysis_details
    #             }
    #         else:
    #             return {
    #                 'success': False,
    #                 'error': 'No pose detected',
    #                 'message': 'Ensure full body is visible in the image'
    #             }
                
        except Exception as e:
            print(f"Error processing image: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_analysis_details(self, landmarks):
        """Get detailed analysis - must be implemented by subclass"""
        raise NotImplementedError("Subclass must implement get_analysis_details method")
    
    def reset_counter(self):
        """Reset rep counter"""
        if self.rep_counter:
            self.rep_counter.reset()
    
    def get_rep_count(self):
        """Get current rep count"""
        return self.rep_counter.rep_count if self.rep_counter else 0
    
    def cleanup(self):
        """Cleanup resources"""
        if self.pose:
            self.pose.close()
        if self.rep_counter:
            self.rep_counter.reset()
        gc.collect()