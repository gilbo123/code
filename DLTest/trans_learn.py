# importing required libraries

from keras.models import Sequential
from scipy.misc import imread
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense
import pandas as pd
import numpy as np

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.preprocessing.image import ImageDataGenerator

from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation

from sklearn.metrics import log_loss



train_data_dir = 'trainingPunnets'
validation_data_dir = 'validatingPunnets'
nb_train_samples = 442 #2000
nb_validation_samples = 58#181

from scipy.misc import imresize




#Load model with static first 8 layers
from keras.models import Model

def vgg16_model(img_rows, img_cols, channel=1, num_classes=None):
    model = VGG16(weights='imagenet', include_top=True)
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    x=Dense(num_classes, activation='softmax')(model.output)
    model=Model(model.input,x)

	#To set the first 8 layers to non-trainable (weights will not be updated)
    for layer in model.layers[:16]:#18??
    	layer.trainable = False

    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model




img_width, img_height = 224, 224 # Resolution of inputs
channel = 3
num_classes = 2
batch_size = 16 
epochs = 10

# Load our model
model = vgg16_model(img_width, img_height, channel, num_classes)

model.summary()


# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model.save_weights('strawberry_try.h5')