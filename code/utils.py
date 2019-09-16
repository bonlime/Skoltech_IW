import time
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