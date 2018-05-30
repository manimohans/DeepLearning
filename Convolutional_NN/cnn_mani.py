#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 15:45:37 2018

@author: mani
"""
"""
#mainly used for image recognition
#needed modules - theano, tensorflow and keras.
#keep this as a template and replace cat/dog pics with 
#pic choice of yours.

It's always good to keep the training set and test set
clearly classified in separate folders

inside the training set and test set, name the files 
by its classification
and with some number at the end.

keras has some good modules to import images

There is no pre-processing needed. We have to do it 
manually in folder
"""

# Part 1 - Building the CNN
#importing the keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D #first step of making CNN
#convolution3d used for videos? time as 3rd dimension?
from keras.layers import MaxPooling2D #max pooling step
from keras.layers import Flatten #convert feature maps 
#to feature vec
from keras.layers import Dense #add fully connected layer 
#and classic ANN

#initializing the CNN
classifier = Sequential() 

# step 1 -> add the convolutional layer
#step 1 convolution 2 max pooling 3 flattening 4 full connection
#we are doing only step 1 here.
#input image + feature detector = feature map ->
#this step is called convolution - creates many feature maps
#this is called the convolution layer with all the f-maps
classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3),
                             activation='relu'))
#common practice - start with 32 feature detectors 
#and then climb up to 64, 128 for 
#the next convolutional layers
#32 feature detectors of dimension 3x3
#input_shape is the dimension of each image, needs to be same
#3 is color channels rgb, 64x64 is dimensions
#for theano back-end it will be 3, 64, 64. channels, dim1, dim2

#step 2 - pooling - we use max-pooling here
#reduce size of feature map and reduce size of fully connected layer

classifier.add(MaxPooling2D(pool_size=(2, 2)))
#most times we take 2x2 pool size.

#adding another conv layer later after running the code once 
#to finetune and improve accuracy of the model
classifier.add(Convolution2D(32, 3, 3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

#step 3 - flattening
#pooling layer needs to be flattened since there are
#many many feature maps in pooling layer
#pooling layer after flattening becomes input layer 
#of the fully connected layer in the next step
classifier.add(Flatten())
#no args needed

#step 4 - full connection - make a classic ANN
#after flattening, we have the input layer
classifier.add(Dense(output_dim = 128, activation = 'relu'))
#this is the hidden layer
#common practice is power of 2, input shape is 64
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))
#this is the output layer
#since it's a binary outcome, use sigmoid activation

#output_dim is units in latest keras lib

#final step - compiling the CNN

classifier.compile(optimizer = 'adam',
                   loss = 'binary_crossentropy', 
                   metrics = ['accuracy'])

#for more than 2 categories, use categorical cross entropy

#part 2 - fitting the CNN to the images
#https://keras.io/preprocessing/image/
#ImageDataGenerator - image augmentation
#we need a lot of image to generalize and find patterns
#rotate, flip, zoom, scale etc. so it's augmented, and 
#there are lots more data to train. All images unique.
#this is crucial to prevent over-fitting. 

#the following is copied from keras documentation
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
#64x64 is the image size we have specified earlier

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)



# Part 3 - Making new predictions

import numpy as np #for image preprocessing
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
#same exact dimensions like we used to train - 64x64
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
#the array has 4 dimensions, so a new dimension needs to be added
#the extra dimension is the batch #, and that's what it expects
#so one dimension in axis 0 (column) is added
result = classifier.predict(test_image)  #predict
training_set.class_indices
#class_indices shows the result dictionary. cats 0 dogs 1
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
    
"""
Evaluation was already made during the training with the validation set, therefore k-Fold Cross Validation is not needed.

Then the techniques to improve and tune a CNN model are the same as for ANNs. So here is the challenge:

Apply the techniques you've learnt and use your architect power to make a CNN that will give you the gold medal:

Bronze medal: Test set accuracy between 80% and 85%

Silver medal: Test set accuracy between 85% and 90%

Gold medal: Test set accuracy over 90%!!

Rules:

- Use the same training set

- Dropout allowed

- Customized SGD allowed

- Specific seeds not allowed


"""

"""
SOLUTION FROM A STUDENT WITH 90+ ACCURACY

from tensorflow.contrib.keras.api.keras.layers import Dropout
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Conv2D
from tensorflow.contrib.keras.api.keras.layers import MaxPooling2D
from tensorflow.contrib.keras.api.keras.layers import Flatten
from tensorflow.contrib.keras.api.keras.layers import Dense
from tensorflow.contrib.keras.api.keras.callbacks import Callback
from tensorflow.contrib.keras.api.keras.preprocessing.image import ImageDataGenerator
from tensorflow.contrib.keras import backend
import os
 
 
class LossHistory(Callback):
    def __init__(self):
        super().__init__()
        self.epoch_id = 0
        self.losses = ''
 
    def on_epoch_end(self, epoch, logs={}):
        self.losses += "Epoch {}: accuracy -> {:.4f}, val_accuracy -> {:.4f}\n"\
            .format(str(self.epoch_id), logs.get('acc'), logs.get('val_acc'))
        self.epoch_id += 1
 
    def on_train_begin(self, logs={}):
        self.losses += 'Training begins...\n'
 
script_dir = os.path.dirname(__file__)
training_set_path = os.path.join(script_dir, '../dataset/training_set')
test_set_path = os.path.join(script_dir, '../dataset/test_set')
 
# Initialising the CNN
classifier = Sequential()
 
# Step 1 - Convolution
input_size = (128, 128)
classifier.add(Conv2D(32, (3, 3), input_shape=(*input_size, 3), activation='relu'))
 
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))  # 2x2 is optimal
 
# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
 
# Adding a third convolutional layer
classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
 
# Step 3 - Flattening
classifier.add(Flatten())
 
# Step 4 - Full connection
classifier.add(Dense(units=64, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units=1, activation='sigmoid'))
 
# Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
 
# Part 2 - Fitting the CNN to the images
batch_size = 32
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
 
test_datagen = ImageDataGenerator(rescale=1. / 255)
 
training_set = train_datagen.flow_from_directory(training_set_path,
                                                 target_size=input_size,
                                                 batch_size=batch_size,
                                                 class_mode='binary')
 
test_set = test_datagen.flow_from_directory(test_set_path,
                                            target_size=input_size,
                                            batch_size=batch_size,
                                            class_mode='binary')
 
# Create a loss history
history = LossHistory()
 
classifier.fit_generator(training_set,
                         steps_per_epoch=8000/batch_size,
                         epochs=90,
                         validation_data=test_set,
                         validation_steps=2000/batch_size,
                         workers=12,
                         max_q_size=100,
                         callbacks=[history])
 
 
# Save model
model_backup_path = os.path.join(script_dir, '../dataset/cat_or_dogs_model.h5')
classifier.save(model_backup_path)
print("Model saved to", model_backup_path)
 
# Save loss history to file
loss_history_path = os.path.join(script_dir, '../loss_history.log')
myFile = open(loss_history_path, 'w+')
myFile.write(history.losses)
myFile.close()
 
backend.clear_session()
print("The model class indices are:", training_set.class_indices)


"""