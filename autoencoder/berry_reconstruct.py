import os
import random
import sys
import cv2
import keras
from matplotlib import pyplot as plt
import numpy as np
import gzip
import copy
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D
from keras.models import Model
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from skimage.measure import compare_ssim
from sklearn.model_selection import train_test_split

class autoencoder():

    def __init__(self, X, Y):
        #sizes
        self.w = X
        self.h = Y
        self.c = 3
        self.cw = int(self.w*0.33)
        self.ch = int(self.h*0.33)

        #test split
        self.train_percent = 0.85

        #input shape
        input_img = Input(shape = (self.h, self.w, self.c))
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
        
        #model
        self.decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(up2) # 28 x 28 x 1
        
        #create ae reference 
        
        self.model = Model(input_img, self.decoded)
        self.model.compile(loss='mean_squared_error', optimizer = RMSprop())
        self.model.summary()
        


    def crop_image_center(self, img):
        '''
        replace the center of the image
        with blank area (black)
        '''
        #make a black square
        square = np.zeros((self.cw, self.ch, self.c))
        #height / width
        x, y = self.w, self.h
        cropx, cropy = self.cw, self.ch
        #crop
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)    
        img[starty:starty+cropy,startx:startx+cropx] = square
        '''
        cv2.imshow('img', img)
        cv2.waitKey()
        cv2.destroyAllWindows()
        '''
        #return
        return img

    def get_data(self, source_path, crop):
        #get files
        files = os.listdir(source_path)
        #shuffle files
        #random.shuffle(files)
        #count = len(files)
        arr = []
        for f in files:
            #read data
            img = cv2.imread(os.path.join(source_path, f))
            img = cv2.resize(img, (self.w, self.h))	
            
            #apply blur
            img = cv2.medianBlur(img, 11)

            #add to array
            if crop:
                #crop image
                c_img = self.crop_image_center(copy.deepcopy(img))
                arr.append(c_img)
                
                ''' 
                cv2.imshow('orig img', img)
                cv2.imshow('cropped img', c_img)
                cv2.waitKey()
                cv2.destroyAllWindows()
                '''
            else:
                #uncropped image
                arr.append(img)

        return np.array(arr)



    def diff_image(self, im1, im2, mode='rgb'):
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





if __name__ == '__main__':

    #data path
    train_path = '/home/gil/Documents/IMAGES/AE/imgs/train/'
    valid_path = '/home/gil/Documents/IMAGES/AE/imgs/valid/'
    test_path = '/home/gil/Documents/IMAGES/AE/imgs/test/'
    out = 'processed/'

    #autoencoder class
    X, Y = 500, 600
    ae = autoencoder(X, Y)
   
    #get data and labels for both train and test
    orig_train = ae.get_data(train_path, False) 
    cropped_train = ae.get_data(train_path, False) 

    # Shapes of training set
    print("\nOriginal training set (images) shape: {}".format(orig_train.shape))
    print("Training set (images) shape: {}".format(cropped_train.shape))


    #normalise
    orig_train = orig_train / np.max(orig_train)
    cropped_train = cropped_train / np.max(cropped_train)
    print(orig_train.dtype, cropped_train.dtype)

    #verify
    print(np.max(orig_train))
    print(np.max(cropped_train))

    #split data
    # X = cropped, y = original
    train_X, valid_X, train_ground, valid_ground = train_test_split(cropped_train,
                                                                 orig_train, 
                                                                 test_size=0.2, 
                                                                 random_state=42)
    
     
    cv2.imshow('cropped img', train_X[0])
    cv2.imshow('orig img', train_ground[0])
    cv2.waitKey()
    cv2.destroyAllWindows()
    
    
    
    #autoencoder params
    BS = 32
    EPOCHS = 1
    CHANNELS = 3
    input_img = Input(shape = (X, Y, CHANNELS))

    autoencoder = ae.model

    # train the convolutional autoencoder
    H = autoencoder.fit(
	train_X, train_ground,
	validation_data=(valid_X, valid_ground),
	epochs=EPOCHS,
	batch_size=BS)



    #plot results
    '''
    loss = H.history['loss']
    val_loss = H.history['val_loss']
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
    

    #remove old data from memory
    del orig_train, cropped_train

    #get test data
    orig_test = ae.get_data(test_path, False)
    cropped_test = ae.get_data(test_path, False)
    
    #normalise cropped data for predictions
    cropped_test = cropped_test / np.max(cropped_test)

    print("\nTest set original shape: {}".format(cropped_test.shape))
    print("Test set cropped shape: {}\n".format(cropped_test.shape))

    print(np.max(orig_test))
    print(np.max(cropped_test))

    ##make predictions
    pred = autoencoder.predict(cropped_test) 
    
    #de-normalise
    cropped_test = cropped_test * 255
    pred = pred * 255



    for i in range(len(pred)):
        #img = cv2.cvtColor(pred[i], cv2.COLOR_BGR2RGB).astype('uint8') 
        p = pred[i].astype('uint8') 
        #img = cv2.resize(img, dsize=(500, 600))
        orig = orig_test[i].astype('uint8')
        print(p.shape, orig.shape)
        print(p.dtype, orig.dtype)
        diff = ae.diff_image(p, orig)
        orig_diff = ae.diff_image(diff, orig)
        p_diff = ae.diff_image(diff, p)
        #hsv_diff = ae.diff_image(p, orig, 'hsv')
         
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

