# exercises/deadlift/RepCounter.py

import numpy as np
import time


class DeadliftRepCounter:
    """Simple rep counter for deadlifts - just hip hinge down and up = 1 rep"""
    
    def __init__(self, form_threshold=0.10):
        self.rep_count = 0
        self.state = "standing"  # States: standing, descending, ascending
        self.form_threshold = form_threshold
        self.form_scores = []
        
        # Hip angle thresholds (shoulder-hip-knee angle)
        self.hip_angle_up = 150      # Standing position
        self.hip_angle_down = 110    # Bent position (hip hinge)
        
        self.min_frames_in_state = 2
        self.frames_in_current_state = 0
        self.deepest_angle = 180
        self.last_rep_time = 0
        self.min_rep_interval = 1.0
        
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        try:
            a = np.array([a.x, a.y])
            b = np.array([b.x, b.y])
            c = np.array([c.x, c.y])
            
            ba = a - b
            bc = c - b
            
            cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
            cosine = np.clip(cosine, -1.0, 1.0)
            angle = np.degrees(np.arccos(cosine))
            
            return angle
        except:
            return 180.0
    
    def calculate_form_score(self, prediction, confidence):
        """Calculate form quality score"""
        if prediction == 'none':
            return 1.0
        elif confidence < 0.6:
            return 0.8
        else:
            return 0.0
    
    def update(self, landmarks, prediction, confidence):
        """Update rep counter - simple down and up = 1 rep"""
        try:
            # Get key landmarks
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            left_knee = landmarks[25]
            right_knee = landmarks[26]
            
            # Calculate hip angles (shoulder-hip-knee)
            left_hip_angle = self.calculate_angle(left_shoulder, left_hip, left_knee)
            right_hip_angle = self.calculate_angle(right_shoulder, right_hip, right_knee)
            avg_hip_angle = (left_hip_angle + right_hip_angle) / 2
            
            # Form tracking
            form_score = self.calculate_form_score(prediction, confidence)
            self.form_scores.append(form_score)
            
            if len(self.form_scores) > 30:
                self.form_scores = self.form_scores[-30:]
            
            rep_counted = False
            state_changed = False
            current_time = time.time()
            
            # SIMPLE 3-STATE MACHINE: standing → descending → ascending → standing (REP!)
            
            if self.state == "standing":
                # Waiting to start going down
                if avg_hip_angle < 140:
                    self.frames_in_current_state += 1
                    if self.frames_in_current_state >= self.min_frames_in_state:
                        self.state = "descending"
                        self.deepest_angle = avg_hip_angle
                        self.form_scores = [form_score]
                        state_changed = True
                        self.frames_in_current_state = 0
                else:
                    self.frames_in_current_state = 0
            
            elif self.state == "descending":
                # Going down - track deepest angle
                if avg_hip_angle < self.deepest_angle:
                    self.deepest_angle = avg_hip_angle
                
                # Check if went low enough AND started coming back up
                if self.deepest_angle <= self.hip_angle_down and avg_hip_angle > self.deepest_angle + 10:
                    self.frames_in_current_state += 1
                    if self.frames_in_current_state >= self.min_frames_in_state:
                        self.state = "ascending"
                        state_changed = True
                        self.frames_in_current_state = 0
                # Reset if didn't go deep enough and coming back up
                elif avg_hip_angle > self.deepest_angle + 20 and self.deepest_angle > self.hip_angle_down:
                    self.state = "standing"
                    self.form_scores = []
                    self.frames_in_current_state = 0
                else:
                    self.frames_in_current_state = 0
            
            elif self.state == "ascending":
                # Coming back up - check if reached standing position
                if avg_hip_angle >= self.hip_angle_up:
                    self.frames_in_current_state += 1
                    if self.frames_in_current_state >= self.min_frames_in_state:
                        # Check time and form quality
                        if current_time - self.last_rep_time >= self.min_rep_interval:
                            avg_form_quality = np.mean(self.form_scores) if self.form_scores else 0
                            
                            # Count rep if form is good (10% tolerance)
                            if avg_form_quality >= (1 - self.form_threshold):
                                self.rep_count += 1
                                rep_counted = True
                                self.last_rep_time = current_time
                        
                        # Reset to standing
                        self.state = "standing"
                        self.deepest_angle = 180
                        self.form_scores = []
                        self.frames_in_current_state = 0
                        state_changed = True
                        
                        return {
                            'rep_count': self.rep_count,
                            'state': self.state,
                            'rep_counted': rep_counted,
                            'form_quality': avg_form_quality if rep_counted else 1.0,
                            'hip_angle': avg_hip_angle,
                            'depth_achieved': self.deepest_angle,
                            'state_changed': state_changed
                        }
                else:
                    self.frames_in_current_state = 0
            
            return {
                'rep_count': self.rep_count,
                'state': self.state,
                'rep_counted': rep_counted,
                'form_quality': np.mean(self.form_scores) if self.form_scores else 1.0,
                'hip_angle': avg_hip_angle,
                'depth_achieved': self.deepest_angle,
                'state_changed': state_changed
            }
            
        except Exception as e:
            print(f"Rep counter error: {e}")
            return {
                'rep_count': self.rep_count,
                'state': self.state,
                'rep_counted': False,
                'form_quality': 1.0,
                'hip_angle': 180.0,
                'depth_achieved': 180.0,
                'state_changed': False
            }
    
    def reset(self):
        """Reset counter"""
        self.rep_count = 0
        self.state = "standing"
        self.form_scores = []
        self.deepest_angle = 180
        self.frames_in_current_state = 0
        self.last_rep_time = 0
 