import cv2 
import time
import imutils
import numpy as np 
from imutils.video import FPS
from imutils.video import VideoStream
from motion_detection import MotionDetector 
from person_detection import SSDDetector, HaarDetector, ObjectTracker
from utils import FPSMeter, VideoRecorder


vs = VideoStream(src=0).start()
time.sleep(2.0)
bg = MotionDetector(mem_time=5)
object_det = SSDDetector()
tracker = ObjectTracker()
frame = vs.read()
recorder = VideoRecorder(h=300, w=400, mem_time=10)
fps = FPSMeter()
idx = 0
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    frame_bg = bg.update(frame)
    recorder.record(frame, bg.has_motion)
    if bg.has_motion:
        detections = object_det.predict(cv2.resize(frame, (300, 300)))
        frame = object_det.draw_predict(frame, detections)
        # if idx%10 == 0: #run detection only every 10th frame
        #     h, w = frame.shape[:2]
        #     _ = tracker.update(frame, detections[:, :4] * [w, h, w, h])
        # else:
        #     pass
    
    
    # else:
    #     detections = tracker.update(frame)
    #     detections = np.array([list(detections) + [1,]])
    # idx += 1
    # # detections = object_det.predict(cv2.resize(frame, (300, 300)))
    # 
    cv2.putText(frame, "FPS: {:.2f}".format(fps.fps), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame,('Not Recording', 'Recording')[bg.has_motion],(10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow('original', frame)
    cv2.imshow('fg', frame_bg)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    fps.update()
    
    #print(fps.fps)

# stop the timer and display FPS information
# print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
# print("[INFO] approx. FPS: {:.2f}".format(fps.fps))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
recorder.close()