import os
import sys
from copy import copy
import cv2
import numpy as np
import matplotlib.pyplot as plt




def power_image(image, factor=2):
    im = np.power(image.astype('float64'), factor)
    return (im/np.power(255, factor-1)).astype('uint8')

def square_image(image):
    img = np.sqrt(image.astype('float64')*255)
    return img.astype('uint8')

def union(a,b):
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0]+a[2], b[0]+b[2]) - x
    h = max(a[1]+a[3], b[1]+b[3]) - y
    return (x, y, w, h)

def preprocess_image(img):
    #HSV
    #hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #h, s, v = cv2.split(hsv)
    #LAB
    L, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
    #get berries
    (T, threshInv2) = cv2.threshold(a, 150, 160, cv2.THRESH_BINARY)
    berries = cv2.bitwise_and(img, img, mask = threshInv2)

    #contours (many)
    (T, threshC) = cv2.threshold(cv2.cvtColor(berries, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(threshC, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #get a bounding rect around all regions
    #greater than x
    rect = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 2000:
            print(area)
            #print(cnt)
            if rect == None:
                rect = cv2.boundingRect(cnt)
            else:
                rect = union(rect, cv2.boundingRect(cnt))


    print(len(contours))

    x, y, w, h = rect[0], rect[1], rect[2], rect[3]
    cropped = img[y:y+h, x:x+w]
    L1, a1, b1 = cv2.split(cv2.cvtColor(cropped, cv2.COLOR_BGR2LAB))

    #hull = cv2.convexHull(contour)

    #get berries again
    (T, threshInv3) = cv2.threshold(a1, 140, 160, cv2.THRESH_BINARY)
    berries2 = cv2.bitwise_and(cropped, cropped, mask = threshInv3)

    #get calyx
    (T, threshInv4) = cv2.threshold(a1, 125, 130, cv2.THRESH_BINARY_INV)
    calyx1 = cv2.bitwise_and(cropped, cropped, mask = threshInv4)

    #union berries and calyx
    berry_and_calyx = cv2.bitwise_or(berries2, calyx1)

    print(berry_and_calyx.dtype)
    #make new image
    new_img = np.full(img.shape, (0,0,0), dtype=np.uint8)
    #new_img = np.zeros(img.shape)
    new_img[y:y+h, x:x+w] = berry_and_calyx


    #plt.hist(b.ravel(),256,[0,256]); plt.show()
    #cv2.imshow('rough', new_img)
    #cv2.imshow('calyx', nowhite)
    #cv2.imshow('a channel', berry_and_calyx)

    return new_img




if __name__ == '__main__':

    input_path = '/home/gil/Documents/IMAGES/ANOMALY/input/'
    output_path = '/home/gil/Documents/IMAGES/ANOMALY/processed/'

    files = os.listdir(input_path)

    for i, f in enumerate(files):
        filepath = input_path + f
        image = cv2.imread(filepath)

        proc_image = preprocess_image(image)
        #x,y,w,h = preprocess_image(image)
        #cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        #cv2.drawContours(img, contours, -1, (0,255,0), 3)

        savepath = output_path + f.split('.jpg')[0] + '_' + str(i) + '.jpg'
        cv2.imwrite(savepath, proc_image)
        '''
        cv2.imshow('image', proc_image)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break
        '''
    #cv2.destroyAllWindows()
