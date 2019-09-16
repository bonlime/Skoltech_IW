import cv2
import numpy as np

PATH_TO_PROTO = 'workdir/mbln_v1_ssd.prototxt'
PATH_TO_MODEL = 'workdir/mbln_v1_ssd.caffemodel'
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
        # predictions has shape (1, 1, 100, 7). The represent 100 most condident predictions
        # 7 numbers are: ??, class idx, confidence, *bbox coords
        return detections

    def draw_predict(self, frame, detections):
        THR = 0.5 # threshold for predictions
        h, w = frame.shape[:2]
        detections = detections[detections[:, 2] > THR]
        for det in detections:
            idx = int(det[1])
            box = det[3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype(int)
            # draw the prediction on the frame
            label = "{}: {:.2f}%".format(CLASSES[idx], det[2] * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
        return frame

# HAARS detector is less robust and works just slighlty faster
PATH_TO_FACE = 'workdir/face_classifier.xml'
PATH_TO_BODY = 'workdir/full_body.xml'
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
