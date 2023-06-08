import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
import math
import os
import random
import string

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
                if conf > 0.9:
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
        lengthOfVideo = 47  # Number of frames needed in the output video
        while True:
            ret, frame = cap.read()
            # if not ret:
            #     print('Error ret')
            #     break
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
    else:
        print("No accident")
    # print('outputed video to ', out_path)
    # out_file = 'Files/output.mp4'
    # video_file = open(out_file, 'rb')


    # out_bytes = video_file.read()
    # return out_bytes

def predictVideo2():
    cap = cv2.VideoCapture('Files/acc2.mp4')
    currentFrame = 0
    model = YOLO('Models/small_80/best.pt')
    while True:
        success, frame = cap.read()
        result = model(frame, stream=True)
        found_accident = False
        for r in result:
            for box in r.boxes:
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                if conf > 0.9:
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
    lengthOfVideo = 119  # Number of frames needed in the output video
    while True:
        ret, frame = cap.read()
        if not ret:
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

if __name__ == '__main__':
    videoFile = 'Files/accidentVideo.mp4'
    predictVideo('Files/acc2.mp4')