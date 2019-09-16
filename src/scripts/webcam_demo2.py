import cv2 
import time
import imutils
import numpy as np 
from imutils.video import FPS
from imutils.video import VideoStream

class MotionDetector:
    """"Stores running median of the backgorund and detects whether or not any motion is present"""
    def __init__(self, mem_size=10, mem_time=10):
        self.i = 0
        self.mem_size = mem_size
        self.mem_time = mem_time
        self.prev = time.time()
        self.initialized = False

    def update(self, image):
        w, h = frame.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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


PATH_TO_PROTO = 'deploy.prototxt'
PATH_TO_MODEL = 'mobilenet_iter_73000.caffemodel'
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(PATH_TO_PROTO, PATH_TO_MODEL)
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

bg = MotionDetector(mem_time=5)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    frame_bg = bg.update(frame)
    # blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843,
	# (300, 300), 127.5)

    cv2.imshow('original', frame)
    cv2.imshow('fg', frame_bg)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    fps.update()
    print(fps.fps())

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()