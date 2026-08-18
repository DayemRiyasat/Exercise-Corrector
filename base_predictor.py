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

    # Share of a rep's frames that may carry a detected fault before the
    # rep as a whole is graded as unclean.
    REP_FAULT_TOLERANCE = 0.35

    def __init__(self, exercise_type):
        self.exercise_type = exercise_type
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.rep_counter = None
        self._rep_predictions = []

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
            self._open_form_gate()
            info = self.rep_counter.update(landmarks, prediction, confidence)
            result['rep_info'] = self._grade_rep(info, prediction)
        return result

    def _open_form_gate(self):
        """Let the counter count every completed rep, not just clean ones.

        Each RepCounter guards its increment with
            if avg_form_quality >= (1 - self.form_threshold)
        where a frame carrying any detected fault scores 0.0. With the
        default threshold of 0.10 that demands near-perfect form, so a rep
        performed with a fault is silently discarded rather than counted.
        That is backwards: whether a rep happened is a question about the
        movement, and whether it was clean is a separate judgement made in
        _grade_rep below.

        Raising the threshold to 1.0 makes the comparison
        `avg_form_quality >= 0.0`, which is always true because form scores
        are never negative. The averaged quality is still reported, so
        nothing is lost.
        """
        counter = self.rep_counter
        if getattr(counter, 'form_threshold', 1.0) < 1.0:
            counter.form_threshold = 1.0

    def _grade_rep(self, info, prediction):
        """Accumulate predictions over the rep in progress, and when one
        completes, label it clean or name the fault that dominated it.

        Grading over the whole rep matters: the frame on which the counter
        finally increments is the moment you return to the start position,
        which is rarely where the fault occurred.
        """
        if not isinstance(info, dict):
            return info

        history = getattr(self, '_rep_predictions', None)
        if history is None:
            history = self._rep_predictions = []
        history.append(prediction)

        if not info.get('rep_counted'):
            return info

        faults = [p for p in history if p not in ('none', 'unknown')]
        # A stray misclassification should not condemn an otherwise clean
        # rep, so judge by the share of faulty frames rather than any single one.
        info['rep_clean'] = (len(faults) / float(len(history))) <= self.REP_FAULT_TOLERANCE
        info['dominant_fault'] = max(set(faults), key=faults.count) if faults else None

        self._rep_predictions = []
        return info

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
        self._rep_predictions = []
        if self.rep_counter:
            self.rep_counter.reset()

    def get_rep_count(self):
        """Get current rep count"""
        return self.rep_counter.rep_count if self.rep_counter else 0

    def cleanup(self):
        """Cleanup resources"""
        self._rep_predictions = []
        if self.rep_counter:
            self.rep_counter.reset()
        gc.collect()