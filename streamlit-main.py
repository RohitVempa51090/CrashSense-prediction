import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from firebase_admin import db
from firebase_admin import storage
import math
import os
import random
import string
import shutil
from datetime import date, datetime
import firebase_admin
from firebase_admin import credentials



def initialize_firebase_app():
    databaseURL = 'https://alertsys-4227d-default-rtdb.firebaseio.com'
    storageURL = 'alertsys-4227d.appspot.com'
    cred = credentials.Certificate('serviceAccountKey.json')
    try:
        firebase = firebase_admin.get_app()
    except ValueError as e:

        firebase = firebase_admin.initialize_app(cred, {
            'databaseURL': databaseURL,
            'storageBucket': storageURL
        })
    return firebase



def predictVideo(videoFile):
    cap = cv2.VideoCapture(videoFile)
    currentFrame = 0
    model = YOLO('Models/small_80/best.pt')
    while True:
        success, frame = cap.read()
        if not success:
            # print('Error success')
            break
        result = model(frame, stream=True)
        found_accident = False
        for r in result:
            for box in r.boxes:
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                if conf > 0.85 and cls != 1:
                    found_accident = True
                    break
            if found_accident:
                break
        if found_accident:
            break

        currentFrame += 1
        cv2.waitKey(0)

    if found_accident:
    # Extracting the necessary frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, currentFrame)
        num = 0
        currentFrame = 0
        lengthOfVideo = 149  # Number of frames needed in the output video
        while True:
            ret, frame = cap.read()
            if not ret:
                # print('Error ret')
                break
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

        cv2_fourcc = cv2.VideoWriter_fourcc(*'avc1')

        frame = cv2.imread(img[0])
        size = list(frame.shape)
        del size[2]
        size.reverse()
        # print(size)

        video = cv2.VideoWriter(out_video_full_path, cv2_fourcc, 30, size)  # output video name, fourcc, fps, size

        for i in range(len(img)):
            video.write(cv2.imread(img[i]))
            print('frame ', i + 1, ' of ', len(img))

        video.release()
        direc_path = 'ImageSequence'
        shutil.rmtree(direc_path)
        os.makedirs(direc_path)
        return 1

    else:
        return 0


def databasePush(videoUrl):
    latitude = 27.1751
    longitude = 78.0421
    current_date = date.today()
    formatted_date = current_date.strftime("%d-%m-%Y")
    current_time = datetime.now().time().strftime("%H:%M:%S")
    place = 'Agra'

    data = {
        'date': formatted_date,
        'time': current_time,
        'place': place,
        'longitude': longitude,
        'latitude': latitude,
        'video': videoUrl
    }
    ref = db.reference('/Accidents/')
    ref.push().set(data)


def videoUpload():
    bucket = storage.bucket()
    video_path = 'Files/output.mp4'
    acc_path = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
    firebase_st_path = f'videos/acc{acc_path}.mp4'
    blob = bucket.blob(firebase_st_path)
    blob.upload_from_filename(video_path)
    blob.make_public()
    video_url = blob.public_url
    databasePush(video_url)


def host():
    st.title("Crash Sense Dashboard")
    st.sidebar.title("Settings")

    st.sidebar.markdown('---')
    video_file = st.sidebar.file_uploader('Upload CCTV Footage', type=['mp4'])


    if video_file:
        with open(os.path.join("Files", video_file.name), 'wb') as f:
            f.write(video_file.getbuffer())

        video = f'Files/{video_file.name}'
        with st.spinner('Processing video...'):
            # Call the predictVideo function and pass the uploaded video file
            res = predictVideo(video)
        if res == 1:
            out_file = 'Files/output.mp4'
            video_file = open(out_file, 'rb')

            out_bytes = video_file.read()
            st.video(out_bytes)
            videoUpload()
        else:
            st.warning("ACCIDENT NOT DETECTED")

    # else:
    #     st.warning("Failed to decode video. Please try again with a valid video file.")


if __name__ == '__main__':
    firebase_instance = initialize_firebase_app()
    try:
        host()
    except SystemExit:
        pass
