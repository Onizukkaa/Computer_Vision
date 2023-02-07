# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 18:36:32 2022

@author: joach
"""

import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

people=["Angelina Jolie","Ben Affleck","Brad Pitt","Elton John"]
#features=np.load("features.npy",allow_pickle=True)
#lables=np.load("labels.npy",allow_pickle=True)

face_recognizer=cv.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_trained.yml")

img=cv.imread(r"D:/programmation/Ecole/Story_24/Validation/Ben Affleck/6.jpg")

gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow("person",gray)


#Detect the face in the image

faces_rect=haar_cascade.detectMultiScale(gray,1.1,4)

for (x,y,w,h) in faces_rect:
    faces_roi=gray[y:y+h,x:x+w]
    
    label,confidence=face_recognizer.predict(faces_roi)
    print(f"label = {people[label]} with a confidence of {confidence}")
    
    cv.putText(img,str(people[label]),(20,20),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),thickness=2)
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)
    
cv.imshow("Detected",img)
cv.waitKey(0)

    