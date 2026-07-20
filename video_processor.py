# video_processor.py - Common video processing class
# Location: project_root/video_processor.py

import cv2
import numpy as np
import base64
import threading
import queue
import time


class VideoProcessor:
    """Threaded video processor for real-time frame processing"""
    
    def __init__(self, predictor):
        self.predictor = predictor
        self.frame_queue = queue.Queue(maxsize=30)
        self.result_queue = queue.Queue(maxsize=30)
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.processing_thread = None
        
    def start(self):
        """Start the video processing thread"""
        self.stop_event.clear()
        self.pause_event.clear()
        self.processing_thread = threading.Thread(target=self._process_frames)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def stop(self):
        """Stop the video processing thread"""
        self.stop_event.set()
        if self.processing_thread:
            self.processing_thread.join(timeout=2)
    
    def pause(self):
        """Pause video processing"""
        self.pause_event.set()
    
    def resume(self):
        """Resume video processing"""
        self.pause_event.clear()
    
    def add_frame(self, frame_data, frame_number, timestamp):
        """Add a frame to the processing queue"""
        try:
            self.frame_queue.put_nowait({
                'data': frame_data,
                'number': frame_number,
                'timestamp': timestamp
            })
            return True
        except queue.Full:
            return False
    
    def get_result(self, timeout=0.1):
        """Get a processed result from the queue"""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def _process_frames(self):
        """Internal method to process frames in a separate thread"""
        while not self.stop_event.is_set():
            # Handle pause
            while self.pause_event.is_set() and not self.stop_event.is_set():
                time.sleep(0.1)
            
            try:
                # Get frame from queue
                frame_info = self.frame_queue.get(timeout=0.1)
                
                # Decode frame
                img_data = base64.b64decode(frame_info['data'].split(',')[1])
                nparr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    continue
                
                # Resize if needed
                height, width = frame.shape[:2]
                if width > 1280:
                    scale = 1280 / width
                    frame = cv2.resize(frame, (1280, int(height * scale)))
                
                # Process frame using predictor
                result = self.predictor.process_frame(frame)
                result.update({
                    'frame_number': frame_info['number'],
                    'timestamp': frame_info['timestamp']
                })
                
                # Add result to queue
                try:
                    self.result_queue.put_nowait(result)
                except queue.Full:
                    # If queue is full, remove oldest and add new
                    try:
                        self.result_queue.get_nowait()
                        self.result_queue.put_nowait(result)
                    except:
                        pass
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing frame: {e}")
                continue