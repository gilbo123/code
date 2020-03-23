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


class autoencoder():

    def __init__(self):
        #arrays
        self.orig_img_arr = []
        self.train_img_arr = []
        self.test_img_arr = []
        
        #sizes
        self.w = 100
        self.h = 100
        self.c = 3
        self.cw = int(self.w*0.33)
        self.ch = int(self.h*0.33)

        #test split
        self.train_percent = 0.85

        #input shape
        input_img = Input(shape = (self.w, self.h, self.c))
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
        '''
        self.model = Model(input_img, decoded)
        self.model.compile(loss='mean_squared_error', optimizer = RMSprop())
        self.model.summary()
        '''


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

    def get_data(self, source_path):
        #get files
        files = os.listdir(source_path)
        #shuffle files
        random.shuffle(files)
        i = 0
        #count = len(files)
        for f in files:
            i+=1
            #read data
            img = cv2.imread(os.path.join(source_path, f))
            img = cv2.resize(img, (self.w, self.h))	
            
            #apply blur
            img = cv2.medianBlur(img, 11)

            #add to array
            if (i/len(files)) < self.train_percent:
                #add the original image to array
                self.orig_img_arr.append(img)
                
                #crop image
                c_img = self.crop_image_center(copy.deepcopy(img))
                self.train_img_arr.append(c_img)

                cv2.imshow('orig img', img)
                cv2.imshow('cropped img', c_img)
                cv2.waitKey()
                cv2.destroyAllWindows()
                
            else:
                #crop image
                c_img = self.crop_image_center(copy.deepcopy(img))
                self.test_img_arr.append(c_img)

        
        return np.array(self.orig_img_arr), np.array(self.train_img_arr), np.array(self.test_img_arr)



if __name__ == '__main__':

    #data path
    path = '/home/gil/Documents/berry_yolo/berry/images/'
    out = 'processed/'

    #autoencoder class
    ae = autoencoder()
   
    '''
    #get data and labels for both train and test
    orig_train, cropped_train, cropped_test = ae.get_data(path)

    # Shapes of training set
    print("Original training set (images) shape: {}".format(orig_train.shape))
    print("Training set (images) shape: {}".format(cropped_train.shape))
    print("Test set (images) shape: {}".format(cropped_test.shape))


    #reshape for network
    #train_data = train_data.reshape(-1, WIDTH, HEIGHT, 3)
    #test_data = test_data.reshape(-1, WIDTH, HEIGHT, 3)
    #print(train_data.shape, test_data.shape)

    #normalise
    orig_train = orig_train / np.max(orig_train)
    cropped_train = cropped_train / np.max(cropped_train)
    cropped_test = cropped_test / np.max(cropped_test)

    #verify
    print(np.max(orig_train))
    print(np.max(cropped_train))
    print(np.max(cropped_test))

    #split data
    # X = cropped, y = original
    train_X, valid_X, train_ground, valid_ground = train_test_split(cropped_train,
                                                                 orig_train, 
                                                                 test_size=0.2, 
                                                                 random_state=42)
    
    cv2.imshow('orig img', train_ground[0])
    cv2.imshow('cropped img', train_X[0])
    cv2.waitKey()
    cv2.destroyAllWindows()
    

    '''
    #autoencoder params
    BATCH_SIZE = 32
    EPOCHS = 1
    CHANNELS = 3
    X, Y = 100, 100
    input_img = Input(shape = (X, Y, CHANNELS))

    #create generators
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0,
            zoom_range=0,
            horizontal_flip=True,
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1)  # randomly shift images vertically (fraction of total height))

    valid_datagen = ImageDataGenerator(rescale=1./255)  


    #TRAIN gen
    train_generator = train_datagen.flow_from_directory(
            '/home/gil/git/code/autoencoder/patches/train',
            target_size=(X, Y),
            class_mode="input",
            batch_size=BATCH_SIZE,
            shuffle=True,
            seed=42
            )

    #VALID gen
    validation_generator = valid_datagen.flow_from_directory(
            '/home/gil/git/code/autoencoder/patches/validation',
            target_size=(X, Y),
            class_mode="input",
            batch_size=BATCH_SIZE,
            shuffle=True,
            seed=42
            )

    #TEST gen
    test_generator = valid_datagen.flow_from_directory(
            '/home/gil/git/code/autoencoder/patches/test',
            target_size=(X, Y),
            class_mode="input",
            batch_size=BATCH_SIZE,
            shuffle=False,
            seed=42
            )


    #create model
    autoencoder = Model(input_img, ae.decoded(input_img))
    autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())

    autoencoder.summary()

    #step size
    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID=validation_generator.n//validation_generator.batch_size
    #Train model
    autoencoder_train = autoencoder.fit_generator(
            train_generator,
            steps_per_epoch=STEP_SIZE_TRAIN,
            epochs=EPOCHS,
            validation_data=validation_generator,
            validation_steps=STEP_SIZE_VALID
            )

    #evaluate
    autoencoder.evaluate_generator(generator=validation_generator, 
            steps=STEP_SIZE_VALID
            )


    #plot results
    loss = autoencoder_train.history['loss']
    val_loss = autoencoder_train.history['val_loss']
    epochs = range(EPOCHS)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


    #make predictions
    STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
    #test_generator.reset()
    pred=autoencoder.predict_generator(test_generator,
            steps=STEP_SIZE_TEST,
            verbose=1
            )

    print(test_generator.n)
    #print(dir(test_generator))

    raw_images = []
    for fp in test_generator.filepaths:
        raw_images.append(cv2.imread(fp))
        

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
        grayA = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
        print(grayA.dtype, grayB.dtype)
        return np.subtract(grayA.astype('uint8'), grayB.astype('uint8'))
        #(score, diff) = compare_ssim(im1, im2, full=True, multichannel=True)
        #print("SSIM: {}".format(score))
        #return (diff * 255).astype("uint8")


    for i in range(len(pred)):
        img = cv2.cvtColor(pred[i], cv2.COLOR_BGR2RGB) 
        img = cv2.resize(img, dsize=(500, 600))
        print(img.shape, raw_images[i].shape)
        diff = diff_image(img, raw_images[i])
        hsv_diff = diff_image(img, raw_images[i], 'hsv')
         
        cv2.namedWindow('Raw', cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow('Raw', 20, 20)
        cv2.namedWindow('RGB diff', cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow('RGB diff', 520, 20)
        cv2.namedWindow('HSV diff', cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow('HSV diff', 1040, 20)
        
        cv2.imshow('Raw', raw_images[i])
        cv2.imshow('RGB diff', diff)
        cv2.imshow('HSV diff', hsv_diff)
        cv2.waitKey()
        cv2.destroyAllWindows()

