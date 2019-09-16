
import shutil
import os
import cv2
import json

# only classes present in COLOR_MAP would be drawn
COLOR_MAP = {
    'person': [0,0,255],
    #'handbag': [255,0,0], 
    #'backpack': [0, 255, 0],
}

def main():
    # remove previous frames
    if os.path.exists('tmp/preds_on_frames'):
        shutil.rmtree('tmp/preds_on_frames/')
    os.mkdir('tmp/preds_on_frames')
    bboxes = json.load(open('bboxes'))
    classes = json.load(open('classes'))
    un_cls = set()
    for cl in classes.values():
        un_cls.update(cl)
    print('ALL CLASSES ON VIDEO: {}'.format(un_cls))
    n_frames = len(os.listdir('tmp/frames'))
    for frame in range(1, n_frames):
        img = cv2.imread('tmp/frames/imgs{:04d}.jpg'.format(frame))
        img_boxes = bboxes[str(frame)]
        img_classes = classes[str(frame)]
        for box, _cls in zip(img_boxes, img_classes):
            if _cls not in COLOR_MAP.keys(): # filter by class
                continue
            cv2.rectangle(img, tuple(box[0]), tuple(box[1]), COLOR_MAP[_cls], 4)
        cv2.imwrite('tmp/preds_on_frames/imgs{:04d}.jpg'.format(frame), img)
        
        


if __name__ == '__main__':
    main()