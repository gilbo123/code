#!/usr/bin/python

import numpy as np
import PyCapture2
from timeit import default_timer as timer


class Capture:
    def __init__(self, camName):
        ''' Constructor for this class. '''
        # Create some members
        self.members = ['Cam1', 'Cam2', 'Cam3', 'Cam4']
        #bus manager
        self.camName = camName
        self.Camera = None
        self.bus = PyCapture2.BusManager()

    '''
    Initializes the Camera
    depending on parameters
    '''
    def initializeCamera(self):
        #create camera objects
        if self.camName == 'RGBTopCamera':
            #RGBTop
            RGBTopCamera = PyCapture2.Camera()
            print(self.bus)
            RGBTopUid = self.bus.getCameraFromIndex(0)
            #RGBTopUid = self.bus.getCameraFromSerialNumber(15435621)
            RGBTopCamera.connect(RGBTopUid)
            RGBTopCamera.startCapture()
            #this camera is this.Camera
            self.Camera = RGBTopCamera
        elif self.camName == 'IRTopCamera':
            #IRTop
            IRTopCamera = PyCapture2.Camera()
            IRTopUid = self.bus.getCameraFromIndex(1)
            IRTopCamera.connect(IRTopUid)
            IRTopCamera.startCapture()
            #this camera is this.Camera
            self.Camera = IRTopCamera
        elif self.camName == 'RGBBtmCamera':
            #RGBBtm
            RGBBtmCamera = PyCapture2.Camera()
            RGBBtmUid = self.bus.getCameraFromSerialNumber(16097126)
            RGBBtmCamera.connect(RGBBtmUid)
            RGBBtmCamera.startCapture()
            #this camera is this.Camera
            self.Camera = RGBBtmCamera
        elif self.camName == 'IRBtmCamera':
            #IRBtm
            IRBtmCamera = PyCapture2.Camera()
            IRBtmUid = self.bus.getCameraFromSerialNumber(18090908)
            IRBtmCamera.connect(IRBtmUid)
            IRBtmCamera.startCapture()
            #this camera is this.Camera
            self.Camera = IRBtmCamera
        else:
            print('Error - cant find camera by name..')



    '''
    Grabs an image
    from the specified camera
    '''
    def grabImage(self):
        start = timer()
        print(self.Camera)
        image = self.Camera.retrieveBuffer()
        cv_image = np.array(image.getData(), dtype="uint8").reshape((image.getRows(), image.getCols()) );
        end = timer()
        time = end - start
        print(time)
        return cv_image
