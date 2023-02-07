# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 12:54:54 2022

@author: joach
"""

import os
import caer
import canaro
import numpy as np
import cv2 as cv
import gc
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical

#%%
IMG_SIZE=(80,80)
channels=1
char_path=(r"D:/programmation/Ecole/Story_24_Computer_Vision/reconnaissance_RN/simpsons_dataset")

#%%
#Ceating a character dic, sorting it on descending order

char_dict={}
for char in os.listdir(char_path):
    char_dict[char]=len(os.listdir(os.path.join(char_path,char)))
    
#sort in decending order
char_dict=caer.sort_dict(char_dict,descending=True)
#%%
print(char_dict)
#%%
#getting the first 10 catégories with the most number of images

characters=[]
count=0

for i in char_dict:
    characters.append(i[0])
    count +=1
    if count>=10:
        break
#%%
print(characters)    
#%%
#Create training data

train=caer.preprocess_from_dir(char_path,characters,channels=channels,IMG_SIZE=IMG_SIZE,isShuffle=True)
#%%
#Visualizing the data 
plt.figure(figsize=(30,30))
plt.imshow(train[0][0],cmap="gray")
plt.show()
#%%
#Separating the array and corresponding labels
featureSet,labels=caer.sep_train(train,IMG_SIZE=IMG_SIZE)
#%%
#Normaize the featureSet =>(0,1)
featureSet=caer.normalize(featureSet)
labels=to_categorical(labels,len(characters))
#%%
x_train,x_val,y_train,y_val=caer.train_val_split(featureSet,labels,val_ratio=0.2)

#%%
del train
del featureSet
del labels
gc.collect()
#%%
BATCH_SIZE=32
EPOCHS=10

#%%
#Image data generator
datagen=canaro.generators.imageDataGenerator()
train_gen=datagen.flow(x_train,y_train,batch_size=BATCH_SIZE)

#%%
#Creating model
model=canaro.models.createSimpsonsModel(IMG_SIZE=IMG_SIZE,channels=channels,output_dim=len(characters),
                                        loss="binary_crossentropy",decay=1e-6,learning_rate=0.001,
                                       momentum=0.9,nesterov=True )
#%%
print(model.summary())
#%%
# Training the model
from tensorflow.keras.callbacks import LearningRateScheduler
#%%
callbacks_list = [LearningRateScheduler(canaro.lr_schedule)]

training = model.fit(train_gen,
                    steps_per_epoch=len(x_train)//BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(x_val,y_val),
                    validation_steps=len(y_val)//BATCH_SIZE,
                    callbacks = callbacks_list)
#%%
test_path=r"D:/programmation/Ecole/Story_24_Computer_Vision/reconnaissance_RN/kaggle_simpson_testset/kaggle_simpson_testset/charles_montgomery_burns_3.jpg"

img=cv.imread(test_path)
plt.imshow(img,cmap="gray")
plt.show()
#%%

def prepare(img):
    img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    img=cv.resize(img,IMG_SIZE)
    img=caer.reshape(img,IMG_SIZE,1)
    return img
#%%

predictions=model.predict(prepare(img))

print(characters[np.argmax(predictions[0])])





























