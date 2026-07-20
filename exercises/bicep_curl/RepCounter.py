# exercises/bicep_curl/RepCounter.py

import numpy as np
import time


class BicepCurlRepCounter:
    """Simple rep counter for bicep curls - tracks elbow flexion and extension"""
    
    def __init__(self, form_threshold=0.10):
        self.rep_count = 0
        self.state = "down"  # States: down, curling, up, lowering
        self.form_threshold = form_threshold
        self.form_scores = []
        
        # Elbow angle thresholds (shoulder-elbow-wrist angle)
        self.elbow_angle_down = 160    # Arm extended (starting position)
        self.elbow_angle_up = 50       # Arm fully curled (bicep contracted)
        
        self.min_frames_in_state = 2
        self.frames_in_current_state = 0
        self.lowest_angle = 180  # Track peak contraction
        self.last_rep_time = 0
        self.min_rep_interval = 0.8  # Bicep curls are faster
        
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
        """Update rep counter - tracks elbow flexion/extension"""
        try:
            # Get key landmarks (use average of both arms)
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_elbow = landmarks[13]
            right_elbow = landmarks[14]
            left_wrist = landmarks[15]
            right_wrist = landmarks[16]
            
            # Calculate elbow angles (shoulder-elbow-wrist)
            left_elbow_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_elbow_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
            avg_elbow_angle = (left_elbow_angle + right_elbow_angle) / 2
            
            # Form tracking
            form_score = self.calculate_form_score(prediction, confidence)
            self.form_scores.append(form_score)
            
            if len(self.form_scores) > 30:
                self.form_scores = self.form_scores[-30:]
            
            rep_counted = False
            state_changed = False
            current_time = time.time()
            
            # SIMPLE STATE MACHINE: down → curling → up → lowering → down (REP!)
            
            if self.state == "down":
                # Arms extended, waiting to start curl
                if avg_elbow_angle < 145:  # Started curling
                    self.frames_in_current_state += 1
                    if self.frames_in_current_state >= self.min_frames_in_state:
                        self.state = "curling"
                        self.lowest_angle = avg_elbow_angle
                        self.form_scores = [form_score]
                        state_changed = True
                        self.frames_in_current_state = 0
                else:
                    self.frames_in_current_state = 0
            
            elif self.state == "curling":
                # Curling up - track lowest angle (peak contraction)
                if avg_elbow_angle < self.lowest_angle:
                    self.lowest_angle = avg_elbow_angle
                
                # Check if reached top (full contraction)
                if avg_elbow_angle <= self.elbow_angle_up:
                    self.frames_in_current_state += 1
                    if self.frames_in_current_state >= self.min_frames_in_state:
                        self.state = "up"
                        state_changed = True
                        self.frames_in_current_state = 0
                # If started lowering before reaching top (incomplete curl)
                elif avg_elbow_angle > self.lowest_angle + 20:
                    # Check if we at least got past halfway
                    if self.lowest_angle <= 90:
                        self.state = "lowering"
                        state_changed = True
                        self.frames_in_current_state = 0
                    else:
                        # Reset if didn't curl enough
                        self.state = "down"
                        self.form_scores = []
                        self.frames_in_current_state = 0
                else:
                    self.frames_in_current_state = 0
            
            elif self.state == "up":
                # At top, waiting to start lowering
                if avg_elbow_angle > self.lowest_angle + 15:  # Started lowering
                    self.frames_in_current_state += 1
                    if self.frames_in_current_state >= self.min_frames_in_state:
                        self.state = "lowering"
                        state_changed = True
                        self.frames_in_current_state = 0
                else:
                    self.frames_in_current_state = 0
            
            elif self.state == "lowering":
                # Lowering back down
                if avg_elbow_angle >= self.elbow_angle_down:  # Back to extended
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
                        
                        # Reset to down state
                        self.state = "down"
                        self.lowest_angle = 180
                        self.form_scores = []
                        self.frames_in_current_state = 0
                        state_changed = True
                        
                        return {
                            'rep_count': self.rep_count,
                            'state': self.state,
                            'rep_counted': rep_counted,
                            'form_quality': avg_form_quality if rep_counted else 1.0,
                            'elbow_angle': avg_elbow_angle,
                            'peak_contraction': self.lowest_angle,
                            'state_changed': state_changed
                        }
                else:
                    self.frames_in_current_state = 0
            
            return {
                'rep_count': self.rep_count,
                'state': self.state,
                'rep_counted': rep_counted,
                'form_quality': np.mean(self.form_scores) if self.form_scores else 1.0,
                'elbow_angle': avg_elbow_angle,
                'peak_contraction': self.lowest_angle,
                'state_changed': state_changed
            }
            
        except Exception as e:
            print(f"Rep counter error: {e}")
            return {
                'rep_count': self.rep_count,
                'state': self.state,
                'rep_counted': False,
                'form_quality': 1.0,
                'elbow_angle': 180.0,
                'peak_contraction': 180.0,
                'state_changed': False
            }
    
    def reset(self):
        """Reset counter"""
        self.rep_count = 0
        self.state = "down"
        self.form_scores = []
        self.lowest_angle = 180
        self.frames_in_current_state = 0
        self.last_rep_time = 0