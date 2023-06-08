# from ultralytics import YOLO
# import cv2

# model = YOLO('best.pt')
# result = model('accident1.jpg', show=True)
# cv2.imshow(result[0].plot())
# cv2.waitKey(0)

from ultralytics import YOLO
import cv2
import cvzone
import math
import os


cap = cv2.VideoCapture('Files/accidentVideo5.mp4')

model = YOLO('Models/medium_80/best.pt')

# className = ['Accident']
className = ['moderate', 'no-accident', 'severe-accident']

def process():
    currentFrame = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        result = model(frame, stream=True)
        for r in result:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),3)
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(frame, (x1, y1, w, h))

                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                if conf>0.86 and cls != 1:
                    cvzone.putTextRect(frame, f'{className[cls]} {conf}', (x1, y1 - 20))
                    # return currentFrame
        currentFrame += 1
        cv2.imshow("Image", frame)
        cv2.waitKey(1)

process()
# frame = process()
#
# cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
# num = 0
# currentFrame = 0
# lengthOfVideo = 47 #Number of frames needed in the output video
# while True:
#     ret, frame = cap.read()
#     cv2.imwrite(f'ImageSequence/{num}.jpg', frame)
#     num += 1
#     if currentFrame == lengthOfVideo:
#         break
#     currentFrame += 1
#     cv2.waitKey(0)

# for r in result:
#     boxes = r.boxes
#     for box in boxes:
#         cls = int(box.cls[0])
#         print(cls)
#
# result = [{ {names: {0: 'car-accident'}, {masks: None} }]

'''
ultralytics.yolo.engine.results.Results object with attributes:

boxes: ultralytics.yolo.engine.results.Boxes object
keypoints: None
keys: ['boxes']
masks: None
names: {0: 'car-accident'}
orig_img: array([[[122, 116, 105],
        [114, 108,  97],
        [115, 109,  98],
        ...,
        [130,  85, 112],
        [133,  88, 115],
        [129,  84, 111]],

       [[117, 111, 100],
        [109, 103,  92],
        [112, 106,  95],
        ...,
        [125,  80, 107],
        [127,  82, 109],
        [122,  77, 104]],

       [[110, 104,  93],
        [104,  98,  87],
        [113, 107,  96],
        ...,
        [124,  79, 106],
        [135,  90, 117],
        [134,  89, 116]],

       ...,

       [[172, 146, 164],
        [165, 139, 157],
        [161, 135, 153],
        ...,
        [127,  91, 107],
        [ 78,  41,  57],
        [ 81,  44,  60]],

       [[165, 139, 157],
        [169, 143, 161],
        [171, 145, 163],
        ...,
        [124,  91, 106],
        [ 75,  39,  55],
        [ 79,  43,  59]],

       [[173, 147, 165],
        [171, 145, 163],
        [169, 143, 161],
        ...,
        [123,  90, 105],
        [ 74,  38,  54],
        [ 79,  43,  59]]], dtype=uint8)
orig_shape: (480, 640)
path: 'C:\\Users\\rohit\\PycharmProjects\\pythonProject\\Files\\test.jpg'
probs: None
speed: {'preprocess': 1.0008811950683594, 'inference': 566.1201477050781, 'postprocess': 1.0066032409667969}
'''