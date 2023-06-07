
"""
Created on Wed Nov 18 13:07:51 2020

@author: win10
"""
# pip install fastapi uvicorn

# 1. Library imports
from ultralytics import YOLO
import cv2
import shutil
import uvicorn  ##ASGI
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse
import math
import os

model = YOLO('Models/large_50/best.pt')

# 2. Create the app object
app = FastAPI()


# 3. Index route, opens automatically on http://127.0.0.1:8000

@app.get('/')
def index():
    return {'message': 'Hello, World'}


# Taking Files
@app.post('/Upload')
async def upload(file: UploadFile = File(...)):
    with open('Files/test.jpg', "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    prediction_result = predictImage()
    image = open('Files/result.jpg', 'rb')

    return StreamingResponse(image, media_type='image/jpeg', headers={'prediction': str(prediction_result), 'Location': 'Dummy_Value'})


@app.post('/Video')
async def video(file: UploadFile = File(...)):
    with open('Files/video.mp4', "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    predictVideo()
    return FileResponse('Files/output.mp4', media_type='video/mp4')


def predictVideo():
    cap = cv2.VideoCapture('Files/video.mp4')
    currentFrame = 0
    while True:
        success, frame = cap.read()
        result = model(frame, stream=True)
        found_accident = False
        for r in result:
            for box in r.boxes:
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                if conf > 0.82:
                    found_accident = True
                    break
            if found_accident:
                break
        if found_accident:
            break

        currentFrame += 1
        cv2.waitKey(0)

    # Extracting the necessary frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, currentFrame)
    num = 0
    currentFrame = 0
    lengthOfVideo = 47  # Number of frames needed in the output video
    while True:
        ret, frame = cap.read()
        cv2.imwrite(f'ImageSequence/{num}.jpg', frame)
        num += 1
        if currentFrame == lengthOfVideo:
            break
        currentFrame += 1
        cv2.waitKey(0)

    # Making a video out of the extracted images
    path = 'ImageSequence/'
    out_path = 'Files/'
    out_video_name = 'output.mp4'
    out_video_full_path = out_path + out_video_name

    pre_imgs = os.listdir(path)
    # To sort the images as per the sequence
    pre_imgs = sorted(pre_imgs, key=lambda x: int(x.split('.')[0]))

    img = []

    for i in pre_imgs:
        i = path + i
        # print(i)
        img.append(i)

    # print(img)

    cv2_fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    frame = cv2.imread(img[0])
    size = list(frame.shape)
    del size[2]
    size.reverse()
    # print(size)

    video = cv2.VideoWriter(out_video_full_path, cv2_fourcc, 24, size)  # output video name, fourcc, fps, size

    for i in range(len(img)):
        video.write(cv2.imread(img[i]))
        print('frame ', i + 1, ' of ', len(img))

    video.release()
    print('outputed video to ', out_path)


def predictImage():
    result = model('Files/test.jpg')
    image = result[0].plot()
    cv2.imwrite('Files/result.jpg', image)
    cls = 1
    for r in result:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
    return cls


# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
# uvicorn main:app --reload

