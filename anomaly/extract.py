import os
import random
import sys
import cv2
import keras
from matplotlib import pyplot as plt
import numpy as np
import gzip
import copy
from tensorflow.keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from skimage.measure import compare_ssim
from sklearn.model_selection import train_test_split

class extractor():

    def __init__(self, X, Y):
        #sizes
        self.w = X
        self.h = Y
        self.c = 3

        #input shape
        input_img = Input(shape = (self.h, self.w, self.c))
        #input = 600 x 500 x 3 (wide and thin)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #480 x 480 x 32
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #240 x 240 x 32
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #240 x 240 x 64
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #120 x 120 x 64
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #120 x 120 x 128 (small and thick)
        '''
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3) #60 x 60 x 128
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3) #60 x 60 x 256 (small and thick)
        #decoder
        conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4) #60 x 60 x 256
        up1 = UpSampling2D(size=(2,2))(conv5) # 120 x 120 x 256
        conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(up1) # 120 x 120 x 128
        '''
        up2 = UpSampling2D(size=(2,2))(conv3) # 240 x 240 x 128
        conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(up2) # 240 x 240 x 64
        up3 = UpSampling2D(size=(2,2))(conv7) # 480 x 480 x 64
        conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(up3) # 480 x 480 x 32

        #model
        self.decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(conv8) # 480 x 480 x 3

        #create ae reference 

        self.model = Model(input_img, self.decoded)
        self.model.compile(optimizer='adam', loss='binary_crossentropy')#loss='mean_squared_error', optimizer = RMSprop())
        self.model.summary()



    def get_data(self, source_path, resize=False):
        #get files
        files = os.listdir(source_path)
        #shuffle files
        #random.shuffle(files)
        #count = len(files)
        arr = []
        for f in files:
            #read data
            img = cv2.imread(os.path.join(source_path, f))
        
            if resize:
                img = cv2.resize(img, (self.w, self.h))

            ''' 
            cv2.imshow('orig img', img)
            cv2.imshow('cropped img', c_img)
            cv2.waitKey()
            cv2.destroyAllWindows()
            '''
            #add to array - normalised
            arr.append(img/np.max(img))

        return np.array(arr)


    '''
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
        grayA = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
        #print(grayA.dtype, grayB.dtype)
        return np.subtract(grayA.astype('uint8'), grayB.astype('uint8'))
        #return np.subtract(im2.astype('uint8'), im1.astype('uint8'))
        #(score, diff) = compare_ssim(im1, im2, full=True, multichannel=True)
        #print("SSIM: {}".format(score))
        #return (diff * 255).astype("uint8")
    '''




if __name__ == '__main__':
    #data path
    '''
    train_orig_path = '../input/'
    train_proc_path = '../processed/'
    test_orig_path = '../test/'
    output_path = '../out/'
    '''
    train_orig_path = '/home/gil/Documents/IMAGES/ANOMALY/sm_input/'
    train_proc_path = '/home/gil/Documents/IMAGES/ANOMALY/sm_proc/'
    test_orig_path = '/home/gil/Documents/IMAGES/ANOMALY/test/'
    output_path = '/home/gil/Documents/IMAGES/ANOMALY/out/'


    #extractor class
    X, Y = 240, 240
    int_model = extractor(X, Y)
    model = int_model.model

    #get data and labels for both train and test
    orig_train = int_model.get_data(train_orig_path, resize=True) 
    #target_train = ae.get_data(train_full_path, resize=True) 
    proc_train = int_model.get_data(train_proc_path, resize=True) 

    # Shapes of training set
    print("\nOriginal training set (images) shape: {}".format(orig_train.shape))
    print("Training set (images) shape: {}".format(proc_train.shape))

    #verify
    print(np.max(orig_train))
    print(np.max(proc_train))

    #split data
    # X = cropped, y = original
    train_X, valid_X, train_ground, valid_ground = train_test_split(proc_train,
                                                                 orig_train,
                                                                 test_size=0.2,
                                                                 random_state=42)
    '''
    cv2.imshow('blurred imgT', train_X[0])
    cv2.imshow('blurred imgV', valid_X[0])
    cv2.imshow('orig imgT', train_ground[0])
    cv2.imshow('orig imgV', valid_ground[0])
    cv2.waitKey()
    cv2.destroyAllWindows()
    '''
    #autoencoder params
    BS = 32
    EPOCHS = 1
    #CHANNELS = 3
    #input_img = Input(shape = (X, Y, CHANNELS))


    # train the convolutional autoencoder
    H = model.fit(
	train_X, train_ground,
	validation_data=(valid_X, valid_ground),
	epochs=EPOCHS,
	batch_size=BS)


    print(H.history.keys())
    model.save('berry_extractor_model')

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
    plt.plot(N, H.history["loss"], label="training loss")
    plt.plot(N, H.history["val_loss"], label="validation loss")
    plt.title("Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig('loss_plot.png')
    

    #remove old data from memory
    del orig_train, proc_train

    #get test data
    orig_test = int_model.get_data(test_orig_path, resize=True)
    #proc_test = int_model.get_data(test_proc_path, resize=True)
    
    print("\nTest set original shape: {}".format(orig_test.shape))

    ##make predictions
    pred = model.predict(orig_test) 

    for i in range(len(pred)):
        cv2.imwrite(output_path + str(i) + '.jpg', pred[i]*255)
        '''
        #img = cv2.cvtColor(pred[i], cv2.COLOR_BGR2RGB).astype('uint8') 
        p = pred[i].astype('uint8') 
        #img = cv2.resize(img, dsize=(500, 600))
        orig = orig_test[i].astype('uint8')
        print(p.shape, orig.shape)
        print(p.dtype, orig.dtype)
        diff = int_model.diff_image(p, cropped_test[i].astype('uint8'))
        #orig_diff = int_model.diff_image(diff, orig)
        #p_diff = int_model.diff_image(diff, p)
        #hsv_diff = int_model.diff_image(p, orig, 'hsv')
        ''' 
        #cv2.namedWindow('Original', cv2.WINDOW_AUTOSIZE)
        #cv2.moveWindow('Original', 20, 20)
        #cv2.namedWindow('Prediction', cv2.WINDOW_AUTOSIZE)
        #cv2.moveWindow('Prediction', 520, 20)
        
        #cv2.imshow('Original', orig_test[i])
        #cv2.imshow('Prediction', pred[i])
        #k = cv2.waitKey()
        #if k == ord('q'):
            #cv2.destroyAllWindows()
            #sys.exit()

