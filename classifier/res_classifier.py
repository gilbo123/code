import os
import random
import sys
import cv2
import keras
from matplotlib import pyplot as plt
import numpy as np
import gzip
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation, Input, Conv2D, MaxPooling2D,UpSampling2D
from keras.models import Model
from keras.optimizers import RMSprop
from keras.utils import np_utils
from keras.constraints import maxnorm
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint

def build_finetune_model(base_model, dropout, fc_layers, num_classes):
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    for fc in fc_layers:
        # New FC layer, random init
        x = Dense(fc, activation='relu')(x) 
        x = Dropout(dropout)(x)

    # New softmax layer
    predictions = Dense(num_classes, activation='softmax')(x) 
    
    finetune_model = Model(inputs=base_model.input, outputs=predictions)

    return finetune_model

def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')
    
    plt.savefig('acc_vs_epochs.png')

    plt.figure()
    plt.plot(epochs, loss, 'r.')
    plt.plot(epochs, val_loss, 'r-')
    plt.title('Training and validation loss')

    plt.show()

def count_images(path):
    count = 0
    folders = os.listdir(path)
    for f in folders:
	files = os.listdir(os.path.join(path,f))
	    count+=len(files)
    return count

#Variables
TRAIN_DIR = '/home/gilbert/Downloads/img'
HEIGHT = 450
WIDTH = 375
BATCH_SIZE = 16 
NUM_EPOCHS = 10
num_train_images = count_images(TRAIN_DIR)

#classes
class_names = os.listdir(TRAIN_DIR)
class_num = len(class_names)

train_datagen =  ImageDataGenerator(
    preprocessing_function=preprocess_input,
    #rotation_range=180,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.25
)

train_generator = train_datagen.flow_from_directory(TRAIN_DIR, 
                                                    target_size=(HEIGHT, WIDTH), 
                                                    batch_size=BATCH_SIZE,
                                                    subset='training') # set as training data


validation_generator = train_datagen.flow_from_directory(TRAIN_DIR, # same directory as training data
                                                    target_size=(HEIGHT, WIDTH),
                                                    batch_size=BATCH_SIZE,
                                                    subset='validation') # set as validation data



#RESNET pre-trained
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))

FC_LAYERS = [256, 256]
dropout = 0.5

finetune_model = build_finetune_model(base_model, 
                                      dropout=dropout, 
                                      fc_layers=FC_LAYERS, 
                                      num_classes=class_num)

#COMPILE
adam = Adam(lr=0.00001)
finetune_model.compile(adam, loss='mean_squared_error', metrics=['accuracy'])

filepath="checkpoints/" + "ResNet50" + "_model_weights.h5"
checkpoint = ModelCheckpoint(filepath, monitor=["acc"], verbose=1, mode='max')
callbacks_list = [checkpoint]

#TRAIN
history = finetune_model.fit_generator(train_generator, epochs=NUM_EPOCHS, workers=8, 
                                      steps_per_epoch = train_generator.samples // BATCH_SIZE,
                                      validation_data = validation_generator, 
                                      validation_steps = validation_generator.samples // BATCH_SIZE, 
                                      shuffle=True, callbacks=callbacks_list)
#Plot results
plot_training(history)
