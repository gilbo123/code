import os
import random
import sys
import cv2
#import keras
from matplotlib import pyplot as plt
import numpy as np


#in_path = '/home/gil/Documents/IMAGES/foreign_tests/'
in_path = '/home/gil/Documents/IMAGES/GAN/pass/'
train_path = '/home/gil/Documents/IMAGES/GAN/TRAIN/'
valid_path = '/home/gil/Documents/IMAGES/GAN/VALID/'


#data split
train_percent = 0.8

#variables
i = 0
X, Y = 500, 600
x_stride, y_stride = 200, 200
#patch_size = 50

#list files
files = os.listdir(in_path)
#shuffle files
random.shuffle(files)
count = len(files)
print('Files found: {}'.format(count))

for f in files:
    i+=1
    #if i > 100:
        #break
    #read data
    img = cv2.imread(os.path.join(in_path, f))
    #img = cv2.resize(img, (X, Y))
    #img.shape == (Y, X, Z)
    #print(img.shape)

    size = max(X, Y)
    #must be rgb image only
    if img.shape != (Y, X, 3):
        #could have ir image (X*2)
        if img.shape == (Y, X*2, 3):
            #if it is then split
            img = img[:,:X]
            img = cv2.resize(img, dsize=(size, size))
        else:
            continue
    else:
        img = cv2.resize(img, dsize=(size, size))

    
    print('Image {} shape: {}'.format(i, img.shape))

    '''
    Extract patches
    '''
    x_cal = int(size/x_stride) #10
    y_cal = int(size/y_stride) #12
    for y in range(y_cal):
        #get the new y values
        y1 = y * y_stride  
        y2 = y1 + y_stride
        for x in range(x_cal):
            #get the new x values
            x1 = x * x_stride
            x2 = x1 + x_stride
            
            print(x1, y1, x2, y2)

            #get the slice variables
            patch = img[y1:y2, x1:x2]
            #print(patch.shape)
            #save the patch
            name = f.split('.jpg')[0] + '_patch_' + str(y) + '-' + str(x) + '.jpg'
            print(name)
            print(i)
            print(count*train_percent)
            
            cv2.imwrite(os.path.join(train_path, name), patch)
            '''
            if i < (count*train_percent):
               cv2.imwrite(os.path.join(train_path, name), patch)
            else:
                cv2.imwrite(os.path.join(valid_path, name), patch)
            '''
