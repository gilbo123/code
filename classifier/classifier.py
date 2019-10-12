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
from keras.applications import ResNet50

def model(input_img):
	#build model
	conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #50 x 50 x 32
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #25 x 25 x 32
	drop1 = Dropout(0.2)(pool1)
	bn1 = BatchNormalization()(drop1)
    
	conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(bn1) #14 x 14 x 64
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
	drop2 = Dropout(0.2)(pool2)
	bn2 = BatchNormalization()(drop2)
    
	conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(bn2) #7 x 7 x 128 (small and thick)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3) #7 x 7 x 64
	drop3 = Dropout(0.2)(pool3)
	bn3 = BatchNormalization()(drop3)
	
	flat1 = Flatten()(bn3)
	drop4 = Dropout(0.2)(flat1)
	
	dense1 = Dense(256, activation='relu', kernel_constraint=maxnorm(3))(drop4)
	drop5 = Dropout(0.2)(dense1)
	bn4 = BatchNormalization()(drop5)
	
	dense2 = Dense(128, activation='relu', kernel_constraint=maxnorm(3))(bn4)
	drop6 = Dropout(0.2)(dense2)
	bn5 = BatchNormalization()(drop6)

	dense3 = Dense(64, activation='relu', kernel_constraint=maxnorm(3))(bn5)
	drop7 = Dropout(0.2)(dense3)
	bn6 = BatchNormalization()(drop7)
	
	output = Dense(class_num, activation='softmax')(bn5)
	return output


def get_data(source_path, test_percent):
	#return arrays
	train_img_arr = []
	train_lbl_arr = []
	test_img_arr = []
	test_lbl_arr = []
	#get class folders
	class_folders = os.listdir(source_path)
	class_index = 0
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
				#train_lbl_arr.append(c)
				#class index instead
				train_lbl_arr.append(class_index)
			else:
				test_img_arr.append(img)
				#add label
				#test_lbl_arr.append(c)
				#class index instead
				test_lbl_arr.append(class_index)
		
		#increment class number
		class_index+=1
	
	#return arrays
	return np.array(train_img_arr), np.array(train_lbl_arr), np.array(test_img_arr), np.array(test_lbl_arr)


WIDTH = 125
HEIGHT = 150


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


#normalise
train_data = train_data / np.max(train_data)
test_data = test_data / np.max(test_data)

#transform labels to one-hot
train_labels = np_utils.to_categorical(train_labels) 
test_labels = np_utils.to_categorical(test_labels)
print(train_labels)
print(test_labels)




#model params
batch_size = 16
class_num = test_labels.shape[1]
epochs = 20
inChannel = 3
input_img = Input(shape = (HEIGHT, WIDTH, inChannel))

 
#create model
m = Model(input_img, model(input_img))
m.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
 
m.summary()

#split train data for validation
from sklearn.model_selection import train_test_split
train_X,valid_X,train_ground,valid_ground = train_test_split(train_data,
                                                             train_labels,
                                                             test_size=0.2,
                                                             random_state=13)

 
m_train = m.fit(train_X, train_ground, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(valid_X, valid_ground)) 

#plot results
loss = m_train.history['loss']
val_loss = m_train.history['val_loss']
epochs = range(epochs)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


'''
#make predictions
pred = autoencoder.predict(test_data)

for i in range(len(test_data)):
	print('Label: {}'.format(test_labels[i]))
	cv2.imshow('Raw', test_data[i])
	cv2.imshow('Reconstruct', pred[i])
	cv2.waitKey()
	cv2.destroyAllWindows()
'''	

