#!/usr/bin/python

import threading
import time
import cv2
import numpy as np
import PyCapture2

from PIL import Image
import matplotlib.pyplot as plt

#unique classes
from Punnet import Punnet
from CamWorker import CamWorker



'''
VARS
'''
winSize = 300
winCorner = 5
borderSpace = 80

windows = {'RGBTop': [winCorner, winCorner], 'IRTop': [winCorner, winCorner + winSize + borderSpace ],
            'RGBBtm': [winCorner + winSize + borderSpace, winCorner], 'IRBtm':[winCorner + winSize + borderSpace, winCorner + winSize + borderSpace]}




'''
INIT
'''
# print('Initializing cameras...')

#create new capture classes for
#each camera (4)
# try:
#     RGBTopCap = Capture('RGBTopCamera')
#     # IRTopCap = Capture('IRTopCamera', bus)
#     # RGBBtmCap = Capture('RGBBtmCamera', bus)
#     # IRBtmCap = Capture('IRBtmCamera', bus)
#     #initialize them
#     RGBTopCap.initializeCamera()
#     # IRTopCap.initializeCamera()
#     # RGBBtmCap.initializeCamera()
#     # IRBtmCap.initializeCamera()
# except PyCapture2.Fc2error as err:
#     print('Failed initialization....')
#     print('Exiting.')
#     RGBTopCap.Camera.disconnect()
#     bus = None


def openWindows():
    for window, pos in windows.items():
        # print(window)
        blank_img = np.zeros((600, 500, 3),dtype=np.uint8)
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, winSize, winSize)
        cv2.imshow(window, blank_img)
        h = pos[0]
        w = pos[1]
        cv2.moveWindow(window, h, w)  # Move it to (40,30
        cv2.waitKey(100)



'''
MAIN LOOP
# '''
if __name__=='__main__':
    #Starting
    # print("Starting.....")
    #open 4 windows for images
    openWindows()
    #thread Lock
    threadLock = threading.Lock()
    #punnet object to
    #hold images and attributes
    punnet = Punnet()

    #set up workers
    RGBTopThread = CamWorker(threadLock, punnet)
    # IRTopThread = CamWorker(threadLock, punnet)
    # RGBBtmThread = CamWorker(threadLock, punnet)
    # IRBtmThread = CamWorker(threadLock, punnet)
    # Start workers
    RGBTopThread.start()
    # IRTopThread.start()
    # RGBBtmThread.start()
    # IRBtmThread.start()

    i=0
    while(i < 1000):
        # res = cv2.resize(punnet.RGBTopImage,(500, 600), interpolation = cv2.INTER_CUBIC)
        threadLock.acquire()
        if(punnet.punnetNeedsDisplaying):

            print(punnet.RGBTopImage.shape)
            cv2.imshow('RGBTop', punnet.RGBTopImage)
            punnet.punnetNeedsDisplaying = False

            cv2.waitKey(10)
            i+=1
        threadLock.release()

    # try:
    #     for i in range(10):
    #         if(RGBTopCap.Camera != None):
    #             print('Capturing.....')
    #             img = RGBTopCap.grabImage()
    #             cv2.imshow('RGB-Top', img)
    #             cv2.waitKey(0)
    #
    # except:
    #     RGBTopCap.Camera.disconnect()
    #     print('Cant capture...')
    #     bus = None
    bus = None
    cv2.destroyAllWindows()
