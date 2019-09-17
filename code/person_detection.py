import cv2
import numpy as np

PATH_TO_PROTO = 'input/mbln_v1_ssd.prototxt'
PATH_TO_MODEL = 'input/mbln_v1_ssd.caffemodel'
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

class SSDDetector:

    def __init__(self):
        print("[INFO] loading model...")
        self.net = cv2.dnn.readNetFromCaffe(PATH_TO_PROTO, PATH_TO_MODEL)

    def predict(self, frame):
        # normalize image
        #frame = np.expand_dims(frame, 0)
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

        # pass the blob through the network and obtain the detections and
        # predictions
        self.net.setInput(blob)
        detections = self.net.forward().squeeze()
        # filter only persons
        detections = detections[detections[:, 1] == 15][:, 2:]
        # change conf, *bbox_coords => *bbox_coords, conf
        detections = np.roll(detections, -1, 1)
        return detections

    def draw_predict(self, frame, detections):
        THR = 0.5 # threshold for predictions
        h, w = frame.shape[:2]
        detections = detections[detections[:, 4] > THR]
        for det in detections:
            box = det[:4] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype(int)
            # draw the prediction on the frame
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0,0,255), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, 'person', (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        return frame

# HAARS detector is less robust and works just slighlty faster
PATH_TO_FACE = 'input/face_classifier.xml'
PATH_TO_BODY = 'input/full_body.xml'
class HaarDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(PATH_TO_FACE)
        self.body_cascade = cv2.CascadeClassifier(PATH_TO_BODY)

    def predict(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(frame) # minNeighbors=5
        bodies = self.body_cascade.detectMultiScale(frame)
        #print(faces)
        return faces, bodies
    
    def draw_predict(self, frame, detections):
        faces, bodies = detections
        for det in faces:
            startX, startY, endX, endY = det[0], det[1], det[0] + det[2], det[1]+det[3]
            # draw the prediction on the frame
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0,0,255), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, 'face', (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        return frame 

class ObjectTracker:
    def __init__(self):
        """Tracks multiple objects on scenes"""
        self.tracker = cv2.MultiTracker_create()
        #self.tracker = cv2.TrackerMOSSE_create() # MOSSE is faster
        #self.tracker = cv2.TrackerKCF_create()

    def update(self, frame, bbox=None):
        if bbox is not None and len(bbox) > 0:
            self.tracker.init(frame, tuple(bbox[0]))
            return bbox
        retval, bbox_out = self.tracker.update(frame)
        if not retval:
            # Tracking has failed
            pass 
        return bbox_out