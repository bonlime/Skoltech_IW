import cv2 
import numpy as np 
import time

class MotionDetector:
    """"Stores running median of the backgorund and detects whether or not any motion is present
        Args:
            mem_size (int): Number of frames to store for memory
            mem_time (int): Number of seconds between memory update"""
    def __init__(self, mem_size=10, mem_time=10):
        self.i = 0
        self.mem_size = mem_size
        self.mem_time = mem_time
        self.prev = time.time()
        self.initialized = False

    def update(self, frame):
        """"Updates `has_motion` attribute and returns background"""
        w, h = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if not self.initialized:
            self.images = [gray for i in range(self.mem_size)]
            self.initialized = True
        if time.time() - self.prev > self.mem_time:
            self.images[self.i] = gray
            self.i = (self.i + 1) % self.mem_size
            self.prev = time.time()
        bg = np.median(self.images, axis=0).astype(np.uint8)
        bg = cv2.absdiff(bg, gray)
        bg = cv2.threshold(bg, 25, 255, cv2.THRESH_BINARY)[1]
        if np.sum(bg > 0) / (w*h) > 0.01: # filter objects less than 1% of the image
            self.has_motion = True
        else:
            self.has_motion = False 
        return bg