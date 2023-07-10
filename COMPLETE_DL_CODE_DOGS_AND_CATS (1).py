#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

datadir = 'D:/PetImages'
categories =['dog','cat']

training_data=[]

def create_training_data():
    for category in categories:
        path = os.path.join(datadir, category)
        
#---------------------- category or label converted to numbers--------------------------------------------------
        class_num = categories.index(category)
    
#-------------------------- images are converted to numbers here----------------------------------------------------
        for img in os.listdir(path):
#-------------------------if some images are broken the process is not stopped---------------------------------------

            try:
#--------------------- here color is not an important factor spo we usegrayscale------------------------------------
                img_array = cv2.imread(os.path.join(path, img),cv2.IMREAD_GRAYSCALE)

#-------------------------------------------resizing operation---------------------------------------------------
                img_size = 50 
                new_array = cv2.resize(img_array, (img_size, img_size))
#  -----------------------------------appending the training data details as a list of list-------------------------
                # training data= [[imgarray, claaificatiton], [....], [......], ..........]
                training_data.append([new_array, class_num])
            except Exception as e:
                pass
            
# ------------------------calling funciton here  to create training data----------------------------------------------
create_training_data()
print(len(training_data))

# SHUFFLING DATA SO THAT MODEL AN LEARN BETTER _ SEE REASON
import random
random.shuffle(training_data)


# loading data into feature and variable
X =[]
y=[]

for features, label in training_data:
    X.append(features)
    y.append(label)
    
# X= np.array(X)
# print(X.shape)
# 1 added because off grayscale

img_size  = 50
X=np.array(X).reshape(-1, img_size,img_size, 1)
# print(X.shape)
print(type(X))


# In[13]:
# NORMALISING

X = X/255.0
X = np.array(X)
y = np.array(y)


# In[14]:


import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D


# In[15]:


# if many models has to be eun on the same machine
# for tensorflow 1 - tf.Session, tf.GPUOptions, tf.COnfigProto
# for 2.0 and above - add compat.v1

# gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction = 0.333)
# sess = tf.compat.v1.Session(config = tf.compat.v1.ConfigProto(gpu_options = gpu_options))


model = Sequential()

# by using X.shape[:1] - we ahve omitted the -1 as it is not needed
# -1 basically means that we can take as many values as we want in for each of the 50*50 matrix
# here it means that it can hold many images - each of size 50*50 and grayscale indicated by 1
# print(X.shape)


# BULDING TWO LAYERS

# first layer
model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size= (2,2)))


# second layer
model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size= (2,2)))
          
model.add(Flatten())
model.add(Dense(64))
          
# output
model.add(Dense(1))          
model.add(Activation('sigmoid'))
          
model.compile(loss = 'binary_crossentropy',
             optimizer = 'adam',
             metrics = ['accuracy'])


# In[16]:


# FITTING DATA

model.fit(X,y, batch_size = 32, epochs = 3, validation_split = 0.1)

model.save('dl_dogs_cats.model')
# In[17]:

categories = ['dog','cat']

def prepare(filepath):
    img_size = 50
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array,(img_size, img_size))
    return new_array.reshape(-1, img_size, img_size, 1)


# model = tf.keras.models.load_model('cnn.model')

prediction = model.predict([prepare('dog.jpg')])
print(categories[int(prediction[0][0])])
prediction = model.predict([prepare('cat.jpg')])
print(categories[int(prediction[0][0])])