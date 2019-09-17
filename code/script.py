# First version of a script. Triggers on motion and starts recording

import cv2
from motion_detection import MotionDetector
from imutils.video import VideoStream

vs = VideoStream(src=0).start()
time.sleep(2.0)
bg = MotionDetector(mem_time=5)


