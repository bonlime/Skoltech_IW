#from face_recognition
import cv2
import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import maskrcnn_resnet50_fpn

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)
model = fasterrcnn_resnet50_fpn(pretrained=True) #maskrcnn_resnet50_fpn(pretrained=True) #
model = model.cuda()
model.eval()
# Load a sample picture and learn how to recognize it.
#obama_image = face_recognition.load_image_file("obama.jpg")
#obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# Load a second sample picture and learn how to recognize it.
#biden_image = face_recognition.load_image_file("biden.jpg")
#biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

# Create arrays of known face encodings and their names
#known_face_encodings = [
#    obama_face_encoding,
#    biden_face_encoding
#]
#known_face_names = [
#    "Barack Obama",
#    "Joe Biden"
#]

# Initialize some variables
# face_locations = []
# face_encodings = []
# face_names = []

process_this_frame = True
PROCESS_EVERY = 50
idx = 0
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
def to_numpy(tensor):
    return tensor.detach().cpu().numpy()

def get_prediction(img):
    THRESHOLD = 0.4
    with torch.no_grad():
        pred = model([img]) # Pass the image to the model
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(to_numpy(pred[0]['labels']))] # Get the Prediction Score
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(to_numpy(pred[0]['boxes']))] # Bounding boxes
    pred_score = list(to_numpy(pred[0]['scores']))
    pred_t = [pred_score.index(x) for x in pred_score if x > THRESHOLD] # Get list of index with score greater than threshold.
    pred_boxes = pred_boxes[:pred_t[-1]+1] if pred_t else []
    pred_class = pred_class[:pred_t[-1]+1] if pred_t else []
    return pred_boxes, pred_class

while True:
    # Grab a single frame of video
    video_capture.get
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, ::-1, ::-1].swapaxes(0,2)
    #print(rgb_small_frame.shape)

    # Only process every other frame of video to save time
    idx += 1
    if idx % PROCESS_EVERY == 0:
        #pass
        inp = torch.from_numpy(rgb_small_frame.copy()).cuda().float()
        pred_boxes, pred_class = get_prediction(inp)
        print(pred_boxes, pred_class)
        # Find all the faces and face encodings in the current frame of video
        #face_locations = face_recognition.face_locations(rgb_small_frame)
        #face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        #face_names = []
        #for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
        #    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        #    name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            # face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            # best_match_index = np.argmin(face_distances)
            # if matches[best_match_index]:
            #     name = known_face_names[best_match_index]

            # face_names.append(name)

    # process_this_frame = not process_this_frame


    # Display the results
    # for (top, right, bottom, left), name in zip(face_locations, face_names):
    #     # Scale back up face locations since the frame we detected in was scaled to 1/4 size
    #     top *= 4
    #     right *= 4
    #     bottom *= 4
    #     left *= 4

    #     # Draw a box around the face
    #     cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    #     # Draw a label with a name below the face
    #     cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
    #     font = cv2.FONT_HERSHEY_DUPLEX
    #     cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    frame = frame[:,::-1, :]
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
