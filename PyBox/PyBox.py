#!/usr/bin/python

import threading
import time
import cv2
from sys import exit
import numpy as np
import PySpin

from PIL import Image
import matplotlib.pyplot as plt

#unique classes
from Punnet import Punnet
from CamWorker import CamWorker



'''
VARS
'''
winHeight = 300 #1200/4
winWidth = 480 #1920/4
winCorner = 5
borderSpace = 40
row_space = 20
col_space = 5


#'WinName':[cols, rows]
windows = {'RGBTop': [winCorner, winCorner], 'IRTop': [winCorner + winWidth + borderSpace + row_space, winCorner ],
            'RGBBtm': [winCorner, winCorner + winHeight + borderSpace + col_space], 'IRBtm':[winCorner + winWidth + 
	    borderSpace + row_space, winCorner + winHeight + borderSpace + col_space]}



'''
INIT WINDOWS
'''
def openWindows():
    for window, pos in windows.items():
        # print(window)
        blank_img = np.zeros((winHeight, winWidth, 3),dtype=np.uint8)
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, winWidth, winHeight)
        cv2.imshow(window, blank_img)
        h = pos[0]
        w = pos[1]
        cv2.moveWindow(window, h, w)  # Move it to (40,30
        cv2.waitKey(100)



'''
MAIN LOOP
# '''
if __name__=='__main__':
    #open 4 windows for images
    openWindows()
    #thread Lock
    threadLock = threading.Lock()
    #punnet object to
    #hold images and attributes
    punnet = Punnet()

    # Retrieve singleton reference to system object
    sys = PySpin.System.GetInstance()
    #print(dir(sys))
    # Retrieve list of cameras from the system
    cam_list = sys.GetCameras()
    #print(cam_list.GetSize())


    #set up workers
    RGBTopThread = CamWorker(threadLock, sys, cam_list, punnet, 'RGBTop')
    RGBTopThread.initCam()
    #IRTopThread = CamWorker(threadLock, sys, cam_list, punnet, 'IRTop')
    #IRTopThread.initCam()
    #RGBBtmThread = CamWorker(threadLock, sys, cam_list, punnet, 'RGBBtm')
    #RGBBtmThread.initCam()
    #IRBtmThread = CamWorker(threadLock, sys, cam_list, punnet, 'IRBtm')
    #IRBtmThread.initCam()
    
    # Start workers
    RGBTopThread.start()
    time.sleep(.05)
    #IRTopThread.start()
    #time.sleep(.05)
    #RGBBtmThread.start()
    #time.sleep(.05)
    #IRBtmThread.start()

    i=0
    while(True):	
        #get a lock on the punnet
        threadLock.acquire()
        if(punnet.punnetNeedsDisplaying):

            print(punnet.RGBTopImage.shape)
            cv2.imshow('RGBTop', punnet.RGBTopImage)
            #cv2.imshow('IRTop', punnet.IRTopImage)
            #cv2.imshow('RGBBtm', punnet.RGBBtmImage)
            #cv2.imshow('IRBtm', punnet.IRBtmImage)
            punnet.punnetNeedsDisplaying = False

            k = cv2.waitKey(33)
            if k==27:    # Esc key to stop
                RGBTopThread.stop()
                RGBTopThread = None
                cv2.destroyAllWindows()
                break

            i+=1
        threadLock.release()
    
    print('Finished')
    exit()
