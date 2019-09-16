import os
import tqdm
import shutil
import argparse
import subprocess
import numpy as np
from PIL import Image
import json
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import maskrcnn_resnet50_fpn

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
    
USE_GPU = True

def get_prediction(img_path):
    THRESHOLD = 0.4
    img = Image.open(img_path) # Load the image
    transform = T.Compose([T.ToTensor()]) # Defing PyTorch Transform
    img = transform(img) # Apply the transform to the image
    img = img.cuda() if USE_GPU else img
    with torch.no_grad():
        pred = model([img]) # Pass the image to the model
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(to_numpy(pred[0]['labels']))] # Get the Prediction Score
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(to_numpy(pred[0]['boxes']))] # Bounding boxes
    pred_score = list(to_numpy(pred[0]['scores']))
    pred_t = [pred_score.index(x) for x in pred_score if x > THRESHOLD] # Get list of index with score greater than threshold.
    pred_boxes = pred_boxes[:pred_t[-1]+1] if pred_t else []
    pred_class = pred_class[:pred_t[-1]+1] if pred_t else []
    return pred_boxes, pred_class

def main(args):
    # 1. Video to frames
    # make sure tmp folder is empty
    if os.path.exists('tmp/frames/'):
        shutil.rmtree('tmp/frames/')
    os.mkdir('tmp/frames')
    # cut video into frames
    subprocess.call(['ffmpeg', '-i', args.inp_video, '-vf', 
                    'fps={}'.format(args.fps), 'tmp/frames/imgs%04d.jpg'])
    n_frames = len(os.listdir('tmp/frames'))
    # 2. Make predictions
    global model
    #model = fasterrcnn_resnet50_fpn(pretrained=True) 
    model = maskrcnn_resnet50_fpn(pretrained=True) #
    model.eval()
    model = model.cuda() if USE_GPU else model
    bboxes = {}
    classes = {}
    for frame in tqdm.trange(1, n_frames):
        pred_bbox, pred_cls = get_prediction('tmp/frames/imgs{:04d}.jpg'.format(frame))
        # convert coords to integers
        bboxes[frame] = np.array(pred_bbox).astype(int).tolist() 
        classes[frame] = pred_cls
    with open('bboxes','w') as f:
        json.dump(bboxes, f)
    with open('classes','w') as f:
        json.dump(classes, f)
    print('DONE')
    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Process video')
    parser.add_argument('inp_video', type=str, help='Path to input video')
    parser.add_argument('--fps', type=int, default=4, help='Frame extraction rate')
    parser.add_argument('--bs', default=1, help='BS to use for processing.')
    args = parser.parse_args()
    main(args)