import os
import random
import sys
import cv2
import keras
from matplotlib import pyplot as plt
import numpy as np
import gzip
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D
from keras.models import Model
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from skimage.measure import compare_ssim


def autoencoder(input_img):
    #encoder
    #input = 50 x 50 x 1 (wide and thin)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #50 x 50 x 32
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #25 x 25 x 32
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small and thick)

    #decoder
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 128
    up1 = UpSampling2D((2,2))(conv4) # 14 x 14 x 128
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1) # 14 x 14 x 64
    up2 = UpSampling2D((2,2))(conv5) # 28 x 28 x 64
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(up2) # 28 x 28 x 1
    return decoded


#preprocessing function
def blur_img(img):
    return cv2.medianBlur(img.astype('uint8'), 11).astype('float32')


#autoencoder params
BATCH_SIZE = 32
EPOCHS = 1
CHANNELS = 3
X, Y = 600, 500
input_img = Input(shape = (X, Y, CHANNELS))

#create generators
train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        preprocessing_function=blur_img
        )

valid_datagen = ImageDataGenerator(rescale=1./255,
        preprocessing_function=blur_img
        )  


#TRAIN gen
train_generator = train_datagen.flow_from_directory(
        '/home/gil/Documents/IMAGES/AE/test_batch/train/',
        target_size=(X, Y),
        class_mode="input",
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42
        )

#VALID gen
validation_generator = valid_datagen.flow_from_directory(
        '/home/gil/Documents/IMAGES/AE/test_batch/valid/',
        target_size=(X, Y),
        class_mode="input",
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42
        )

#TEST gen
test_generator = valid_datagen.flow_from_directory(
        '/home/gil/Documents/IMAGES/AE/test_batch/test/',
        target_size=(X, Y),
        class_mode="input",
        batch_size=BATCH_SIZE,
        shuffle=False,
        seed=42
        )


#create model
autoencoder = Model(input_img, autoencoder(input_img))
autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())

autoencoder.summary()

#step size
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=validation_generator.n//validation_generator.batch_size
#Train model
H = autoencoder.fit_generator(
        train_generator,
        steps_per_epoch=STEP_SIZE_TRAIN,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=STEP_SIZE_VALID
        )

autoencoder.save_weights("model.h5")

#evaluate
autoencoder.evaluate_generator(generator=validation_generator, 
        steps=STEP_SIZE_VALID
        )


#plot results
'''
loss = autoencoder_train.history['loss']
val_loss = autoencoder_train.history['val_loss']
epochs = range(EPOCHS)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
'''

N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig('loss_plot.png')
    

#make predictions
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
#test_generator.reset()
pred=autoencoder.predict_generator(test_generator,
        steps=STEP_SIZE_TEST,
        verbose=1
        )

print(test_generator.n)
#print(dir(test_generator))

#de-normalise
pred = pred * 255

raw_images = []
for fp in test_generator.filepaths:
    im = cv2.imread(fp)
    raw_images.append(blur_img(im))
    

def diff_image(im1, im2, mode='rgb'):
    #convert to hsv, if required
    if 'hsv' in mode:
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2HSV)
        h, s1,v = cv2.split(im1)
        im1 = cv2.merge((s1, s1, s1))
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2HSV)
        h, s2,v = cv2.split(im2)
        im2 = cv2.merge((s2, s2, s2))

    # convert the images to grayscale
    #grayA = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    #grayB = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    #print(grayA.dtype, grayB.dtype)
    #return np.subtract(grayA.astype('uint8'), grayB.astype('uint8'))
    return np.subtract(im2.astype('uint8'), im1.astype('uint8'))
    #(score, diff) = compare_ssim(im1, im2, full=True, multichannel=True)
    #print("SSIM: {}".format(score))
    #return (diff * 255).astype("uint8")

for i in range(len(pred)):
    p = cv2.cvtColor(pred[i], cv2.COLOR_BGR2RGB).astype('uint8') 
    #p = pred[i].astype('uint8') 
    #img = cv2.resize(img, dsize=(500, 600))
    orig = raw_images[i].astype('uint8')
    print(p.shape, orig.shape)
    print(p.dtype, orig.dtype)
    diff = diff_image(p, orig)
    orig_diff = diff_image(diff, orig)
    p_diff = diff_image(diff, p)
    #hsv_diff = diff_image(p, orig, 'hsv')
     
    cv2.namedWindow('Original', cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow('Original', 20, 20)
    cv2.namedWindow('Prediction', cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow('Prediction', 320, 20)
    cv2.namedWindow('RGB diff', cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow('RGB diff', 520, 20)
    #cv2.namedWindow('orig diff', cv2.WINDOW_AUTOSIZE)
    #cv2.moveWindow('orig diff', 720, 20)
    cv2.namedWindow('p diff', cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow('p diff', 920, 20)
    
    cv2.imshow('Original', orig)
    cv2.imshow('Prediction', p)
    cv2.imshow('RGB diff', diff)
    #cv2.imshow('orig diff', orig_diff)
    cv2.imshow('p diff', p_diff)
    cv2.waitKey()
    cv2.destroyAllWindows()
