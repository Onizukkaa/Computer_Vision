# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 16:50:30 2022

@author: joach
"""

import os
import cv2 as cv
import numpy as np

people=["Angelina Jolie","Ben Affleck","Brad Pitt","Elton John"]
DIR=r'D:\programmation\Ecole\Story_24'
haar_cascade = cv.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

features=[]
labels=[]

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path,img)

            img_array = cv.imread(img_path)
            if img_array is None:
                continue 
                
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)
create_train()
features=np.array(features,dtype="object")
labels=np.array(labels)

face_recognizer=cv.face.LBPHFaceRecognizer_create()

#Train the recoginser on the features list en the labels
face_recognizer.train(features,labels)

face_recognizer.save("face_traned.yml")
np.save("features.npy",features)
np.save("labels.npy",labels)