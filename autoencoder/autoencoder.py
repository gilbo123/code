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

def get_data(source_path, test_percent):
	#return arrays
	train_img_arr = []
	train_lbl_arr = []
	test_img_arr = []
	test_lbl_arr = []
	#get class folders
	class_folders = os.listdir(source_path)
	for c in class_folders:
		#join paths
		c_path = os.path.join(source_path, c)
		#get files
		files = os.listdir(c_path)
		#shuffle files
		random.shuffle(files)
		i = 0
		count = len(files)
		for f in files:
			i+=1
			#read data
			img = cv2.imread(os.path.join(c_path, f))
			img = cv2.resize(img, (WIDTH, HEIGHT))	
			#add to array
			if (i/count) < test_percent:
				train_img_arr.append(img)
				#add label
				train_lbl_arr.append(c)
			else:
				test_img_arr.append(img)
				#add label
				test_lbl_arr.append(c)

	
	return np.array(train_img_arr), np.array(train_lbl_arr), np.array(test_img_arr), np.array(test_lbl_arr)


WIDTH = 60
HEIGHT = 60


#data path
path = '/home/gilbert/Downloads/img'
#get data and labels for both train and test
test_percent = 0.85
train_data, train_labels, test_data, test_labels = get_data(path, test_percent)

# Shapes of training set
print("Training set (images) shape: {shape}".format(shape=train_data.shape))
print("Test set (images) shape: {shape}".format(shape=test_data.shape))

# Shapes of test set
print("Train set (labels) shape: {shape}".format(shape=train_labels.shape))
print("Test set (labels) shape: {shape}".format(shape=test_labels.shape))



#reshape for network
train_data = train_data.reshape(-1, WIDTH, HEIGHT, 3)
test_data = test_data.reshape(-1, WIDTH, HEIGHT, 3)
print(train_data.shape, test_data.shape)

#normalise
train_data = train_data / np.max(train_data)
test_data = test_data / np.max(test_data)

#verify
print(np.max(train_data))
print(np.max(test_data))


#split data
from sklearn.model_selection import train_test_split
train_X,valid_X,train_ground,valid_ground = train_test_split(train_data,
                                                             train_data, 
                                                             test_size=0.2, 
                                                             random_state=13)

#autoencoder
batch_size = 8
epochs = 1
inChannel = 3
x, y = WIDTH, HEIGHT
input_img = Input(shape = (x, y, inChannel))

#create model
autoencoder = Model(input_img, autoencoder(input_img))
autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())

#autoencoder.summary()

autoencoder_train = autoencoder.fit(train_X, train_ground, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_ground))



#plot results
loss = autoencoder_train.history['loss']
val_loss = autoencoder_train.history['val_loss']
epochs = range(epochs)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


#make predictions
pred = autoencoder.predict(test_data)

for i in range(len(test_data)):
	print('Label: {}'.format(test_labels[i]))
	cv2.imshow('Raw', test_data[i])
	cv2.imshow('Reconstruct', pred[i])
	cv2.waitKey()
	cv2.destroyAllWindows()
	

