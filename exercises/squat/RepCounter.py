
import numpy as np
import time


class SquatRepCounter:
    """Advanced rep counter with landmark-based tracking for squats"""
    
    def __init__(self, form_threshold=0.15):
        self.rep_count = 0
        self.state = "standing"
        self.form_threshold = form_threshold
        self.form_scores = []
        self.knee_angle_down = 100
        self.knee_angle_up = 150
        self.hip_below_knee_threshold = 0.0
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
    
    def check_squat_depth(self, hip, knee):
        """Check if hip is at or below knee"""
        return (hip.y - knee.y) >= self.hip_below_knee_threshold
    
    def calculate_form_score(self, prediction, confidence):
        """Calculate form score"""
        if prediction == 'none':
            return 1.0
        elif confidence < 0.6:
            return 0.8
        else:
            return 0.0
    
    def update(self, landmarks, prediction, confidence):
        """Update rep counter"""
        try:
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            left_knee = landmarks[25]
            right_knee = landmarks[26]
            left_ankle = landmarks[27]
            right_ankle = landmarks[28]
            
            left_knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
            avg_knee_angle = (left_knee_angle + right_knee_angle) / 2
            
            left_depth_ok = self.check_squat_depth(left_hip, left_knee)
            right_depth_ok = self.check_squat_depth(right_hip, right_knee)
            depth_ok = left_depth_ok or right_depth_ok
            
            form_score = self.calculate_form_score(prediction, confidence)
            self.form_scores.append(form_score)
            
            if len(self.form_scores) > 30:
                self.form_scores = self.form_scores[-30:]
            
            rep_counted = False
            state_changed = False
            current_time = time.time()
            
            if self.state == "standing":
                if avg_knee_angle < 135:
                    self.frames_in_current_state += 1
                    if self.frames_in_current_state >= self.min_frames_in_state:
                        self.state = "descending"
                        self.deepest_angle = avg_knee_angle
                        self.form_scores = [form_score]
                        state_changed = True
                        self.frames_in_current_state = 0
                else:
                    self.frames_in_current_state = 0
            
            elif self.state == "descending":
                if avg_knee_angle < self.deepest_angle:
                    self.deepest_angle = avg_knee_angle
                
                if avg_knee_angle <= self.knee_angle_down and depth_ok:
                    self.frames_in_current_state += 1
                    if self.frames_in_current_state >= self.min_frames_in_state:
                        self.state = "bottom"
                        state_changed = True
                        self.frames_in_current_state = 0
                elif avg_knee_angle > self.deepest_angle + 15:
                    self.state = "standing"
                    self.form_scores = []
                    self.frames_in_current_state = 0
                else:
                    self.frames_in_current_state = 0
            
            elif self.state == "bottom":
                if avg_knee_angle > self.deepest_angle + 10:
                    self.frames_in_current_state += 1
                    if self.frames_in_current_state >= self.min_frames_in_state:
                        self.state = "ascending"
                        state_changed = True
                        self.frames_in_current_state = 0
                else:
                    self.frames_in_current_state = 0
            
            elif self.state == "ascending":
                if avg_knee_angle >= self.knee_angle_up:
                    self.frames_in_current_state += 1
                    if self.frames_in_current_state >= self.min_frames_in_state:
                        if current_time - self.last_rep_time >= self.min_rep_interval:
                            avg_form_quality = np.mean(self.form_scores) if self.form_scores else 0
                            
                            if avg_form_quality >= (1 - self.form_threshold):
                                self.rep_count += 1
                                rep_counted = True
                                self.last_rep_time = current_time
                        
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
                            'knee_angle': avg_knee_angle,
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
                'knee_angle': avg_knee_angle,
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
                'knee_angle': 180.0,
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