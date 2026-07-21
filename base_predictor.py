# base_predictor.py - Common base class for all exercises
# Location: project_root/base_predictor.py
#
# Landmark-based version. Pose detection now happens in the browser
# using MediaPipe Tasks for JavaScript. This module receives the 33
# landmarks the browser already extracted and runs the trained
# classification model on them. No mediapipe, cv2, or threading here,
# so it runs cleanly inside a PythonAnywhere web worker.

import numpy as np
import tensorflow as tf
import joblib
import os
import gc


class ExercisePredictor:
    """Base class for all exercise predictors"""

    def __init__(self, exercise_type):
        self.exercise_type = exercise_type
        self.model = None
        self.scaler = None
        self.label_encoder = None
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

            print(f"Model loaded successfully: {self.exercise_type}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
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
        """Extract features from landmarks - implemented by subclass"""
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

    def analyze_landmarks(self, landmarks, count_rep=True):
        """Run prediction, and optionally rep counting, on landmarks
        that were extracted in the browser. Replaces process_frame."""
        prediction, confidence = self.predict(landmarks)
        result = {
            'success': True,
            'prediction': prediction,
            'confidence': confidence
        }
        if count_rep and self.rep_counter is not None:
            result['rep_info'] = self.rep_counter.update(landmarks, prediction, confidence)
        return result

    def analyze_image_landmarks(self, landmarks):
        """Prediction plus detailed analysis for a single still image.
        Does not advance the rep counter, matching the original image path."""
        prediction, confidence = self.predict(landmarks)
        analysis_details = self.get_analysis_details(landmarks)
        return {
            'success': True,
            'prediction': prediction,
            'confidence': confidence,
            'analysis_details': analysis_details
        }

    def get_analysis_details(self, landmarks):
        """Get detailed analysis - implemented by subclass"""
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
        if self.rep_counter:
            self.rep_counter.reset()
        gc.collect()