import time
import cv2
import os
class FPSMeter:
    """Computes and stores the average and current value
        Attributes:
            val - last value
            avg - true average
            avg_smooth - smoothed average"""
    def __init__(self, reset_every=100):
        self._reset_every = reset_every
        self.reset()

    def reset(self):
        self._start = time.time()
        self._end = None
        self._numFrames = 0

    def update(self):
        self._numFrames += 1
        if self._numFrames > self._reset_every:
            self.reset()
    
    @property
    def fps(self):
        return self._numFrames / (time.time() - self._start) 

class VideoRecorder:
    def __init__(self, h, w, root_dir='workdir/output/clips/', mem_time=5):
        """Convinience wrapper around cv2.VideoWriter object
        Args:
            w (int): frame width
            h (int): frame height
            root_dir (str): path where to put clips
            mem_times (int): number of frames to wait before ending the clip
            """
        os.makedirs(root_dir, exist_ok=True)
        self.root_dir = root_dir
        self.h = h
        self.w = w
        self.mem_time = mem_time
        self.vid = None

    def record(self, frame, motion=True):
        if motion and self.vid: # already recording
            self.vid.write(frame)
            self.since_last = 0
        elif motion: # start recording 
            self._init()
        elif self.vid: # no motion but still recording
            self.since_last += 1
            if self.since_last < self.mem_time:
                self.vid.write(frame)
            elif self.since_last == self.mem_time:
                self.close()

    def close(self):
        if self.vid is not None:
            self.vid.release()
            self.vid = None

    def _init(self):
        self.since_last = 0
        filename = os.path.join(self.root_dir, '{}.avi'.format(int(time.time())))
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        self.vid = cv2.VideoWriter(filename, fourcc, 25, (self.w, self.h))
