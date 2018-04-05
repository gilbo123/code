from keras.models import Sequential
from scipy.misc import imread
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense
import pandas as pd
import numpy as np
import argparse
import imutils
import cv2
import os

from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.models import Model
from keras.optimizers import SGD
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense

from sklearn.metrics import log_loss

# # dimensions of our images.
# img_width, img_height = 500, 600#150, 150

# input_shape = (img_width, img_height, 3)

#font for printing result on image
font = cv2.FONT_HERSHEY_SIMPLEX

#Image type
imagetype = 'RGBTop'

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_dir", required=True,
	help="path to input image directory")
args = vars(ap.parse_args())


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



model.load_weights("strawberry_try.h5")

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print("Loaded model from disk")


count = 0
subdirs = [x[0] for x in os.walk(args["input_dir"])]                                                                            
for subdir in subdirs:                                                                                            
	files = os.walk(subdir).next()[2]
	if (len(files) > 0):
		for file in files:
			if (imagetype in file):
				print os.path.join(subdir, file)
				img = cv2.imread(os.path.join(subdir, file), -1) #-1 no change to image

				#image to display
				small = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
				crop_img = small[0:600, 300:800] # Crop from x, y, w, h -> 100, 200, 800, 1200

				#image to process
				resized_image = cv2.resize(crop_img, (224, 224)) 
				# evaluate loaded model on test data
				X = np.array(resized_image)
				X = resized_image.reshape((1, img_width, img_height, channel))
				score = model.predict(X)
				print(score[0])

				cv2.putText(crop_img, np.array2string(score[0], formatter={'float_kind':lambda x: "%.2f" % x}),(100,100), font, 2,(255,255,255),2,cv2.LINE_AA)
				cv2.imshow('image',crop_img)
				


				k = cv2.waitKey(0)
				if k != ord('x'): # wait for 's' key to save and exit
					continue
				else:
					cv2.destroyAllWindows()
					break
				#make a "back button"
				#make a "Save to temp folder button"
				#make a "skip button"
				#add a count so I know when it ENDSSS!!!!

				if k == ord('x'):
					break #break out of the outer loop too

				count+=1


cv2.destroyAllWindows()