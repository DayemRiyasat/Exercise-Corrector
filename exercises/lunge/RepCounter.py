import numpy as np
import time


class LungeRepCounter:
    """Advanced rep counter with landmark-based tracking for lunges"""
    
    def __init__(self, form_threshold=0.10):
        self.rep_count = 0
        self.state = "standing"
        self.form_threshold = form_threshold
        self.form_scores = []
        
        # Lunge-specific thresholds
        self.front_knee_angle_down = 100  # Front knee should bend to ~90°
        self.front_knee_angle_up = 150    # Front knee straightens
        self.back_knee_angle_down = 110   # Back knee bends during lunge
        
        self.min_frames_in_state = 2
        self.frames_in_current_state = 0
        self.deepest_front_angle = 180
        self.last_rep_time = 0
        self.min_rep_interval = 1.0
        
        # Lunge side tracking (which leg is forward)
        self.active_leg = None  # 'left' or 'right'
        
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
    
    def detect_forward_leg(self, left_ankle, right_ankle):
        """Detect which leg is forward based on ankle positions"""
        # In camera view, lower Y value means higher in frame (further back)
        # Higher Y value means lower in frame (closer/forward)
        
        if abs(left_ankle.y - right_ankle.y) > 0.05:  # Significant difference
            if left_ankle.y > right_ankle.y:
                return 'left'  # Left leg is forward (lower in frame)
            else:
                return 'right'  # Right leg is forward
        return None  # Neutral stance
    
    def calculate_form_score(self, prediction, confidence):
        """Calculate form score"""
        if prediction == 'none':
            return 1.0
        elif confidence < 0.6:
            return 0.8
        else:
            return 0.0
    
    def update(self, landmarks, prediction, confidence):
        """Update rep counter for lunges"""
        try:
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            left_knee = landmarks[25]
            right_knee = landmarks[26]
            left_ankle = landmarks[27]
            right_ankle = landmarks[28]
            
            # Detect which leg is forward
            forward_leg = self.detect_forward_leg(left_ankle, right_ankle)
            
            # If no clear forward leg, maintain previous or default to left
            if forward_leg is None:
                forward_leg = self.active_leg if self.active_leg else 'left'
            else:
                self.active_leg = forward_leg
            
            # Calculate angles based on forward leg
            if forward_leg == 'left':
                front_knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
                back_knee_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
            else:
                front_knee_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
                back_knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
            
            # Calculate form score
            form_score = self.calculate_form_score(prediction, confidence)
            self.form_scores.append(form_score)
            
            if len(self.form_scores) > 30:
                self.form_scores = self.form_scores[-30:]
            
            rep_counted = False
            state_changed = False
            current_time = time.time()
            
            # State machine for lunge rep counting
            if self.state == "standing":
                # Detect descent when front knee starts bending significantly
                if front_knee_angle < 135:
                    self.frames_in_current_state += 1
                    if self.frames_in_current_state >= self.min_frames_in_state:
                        self.state = "descending"
                        self.deepest_front_angle = front_knee_angle
                        self.form_scores = [form_score]
                        state_changed = True
                        self.frames_in_current_state = 0
                else:
                    self.frames_in_current_state = 0
            
            elif self.state == "descending":
                # Track deepest point
                if front_knee_angle < self.deepest_front_angle:
                    self.deepest_front_angle = front_knee_angle
                
                # Reached bottom when front knee is sufficiently bent
                if front_knee_angle <= self.front_knee_angle_down:
                    self.frames_in_current_state += 1
                    if self.frames_in_current_state >= self.min_frames_in_state:
                        self.state = "bottom"
                        state_changed = True
                        self.frames_in_current_state = 0
                # Detect if user started ascending without reaching proper depth
                elif front_knee_angle > self.deepest_front_angle + 15:
                    self.state = "standing"
                    self.form_scores = []
                    self.frames_in_current_state = 0
                else:
                    self.frames_in_current_state = 0
            
            elif self.state == "bottom":
                # Detect start of ascent
                if front_knee_angle > self.deepest_front_angle + 10:
                    self.frames_in_current_state += 1
                    if self.frames_in_current_state >= self.min_frames_in_state:
                        self.state = "ascending"
                        state_changed = True
                        self.frames_in_current_state = 0
                else:
                    self.frames_in_current_state = 0
            
            elif self.state == "ascending":
                # Complete rep when front knee straightens
                if front_knee_angle >= self.front_knee_angle_up:
                    self.frames_in_current_state += 1
                    if self.frames_in_current_state >= self.min_frames_in_state:
                        if current_time - self.last_rep_time >= self.min_rep_interval:
                            avg_form_quality = np.mean(self.form_scores) if self.form_scores else 0
                            
                            # Count rep if form quality is acceptable
                            if avg_form_quality >= (1 - self.form_threshold):
                                self.rep_count += 1
                                rep_counted = True
                                self.last_rep_time = current_time
                        
                        self.state = "standing"
                        self.deepest_front_angle = 180
                        self.form_scores = []
                        self.frames_in_current_state = 0
                        state_changed = True
                        
                        return {
                            'rep_count': self.rep_count,
                            'state': self.state,
                            'rep_counted': rep_counted,
                            'form_quality': avg_form_quality if rep_counted else 1.0,
                            'front_knee_angle': front_knee_angle,
                            'back_knee_angle': back_knee_angle,
                            'depth_achieved': self.deepest_front_angle,
                            'active_leg': forward_leg,
                            'state_changed': state_changed
                        }
                else:
                    self.frames_in_current_state = 0
            
            return {
                'rep_count': self.rep_count,
                'state': self.state,
                'rep_counted': rep_counted,
                'form_quality': np.mean(self.form_scores) if self.form_scores else 1.0,
                'front_knee_angle': front_knee_angle,
                'back_knee_angle': back_knee_angle,
                'depth_achieved': self.deepest_front_angle,
                'active_leg': forward_leg,
                'state_changed': state_changed
            }
            
        except Exception as e:
            print(f"Rep counter error: {e}")
            return {
                'rep_count': self.rep_count,
                'state': self.state,
                'rep_counted': False,
                'form_quality': 1.0,
                'front_knee_angle': 180.0,
                'back_knee_angle': 180.0,
                'depth_achieved': 180.0,
                'active_leg': 'left',
                'state_changed': False
            }
    
    def reset(self):
        """Reset counter"""
        self.rep_count = 0
        self.state = "standing"
        self.form_scores = []
        self.deepest_front_angle = 180
        self.frames_in_current_state = 0
        self.last_rep_time = 0
        self.active_leg = None