import threading
from sys import exit
import numpy as np
import cv2
import time
import PyCapture2
from timeit import default_timer as timer

threadLock = threading.Lock()


class CamWorker(threading.Thread):
    # Our workers constructor, note the super() 
    # method which is vital if we want this
    # to function properly
    def __init__(self, lock, punnet):
        super(CamWorker, self).__init__()
        self.blank_img = np.zeros((600, 500, 3),dtype=np.uint8)
        self.cv_image = cv2.imread('/home/gilbert/Documents/code/'+
				'openCV/transfer_learn/berries/pass/'+
				'20180427_220416_0.jpg')
        self.lock = lock
        self.punnet = punnet
        self.RGBTopCamera = None
        #camera set up - RGBTop
        self.bus = PyCapture2.BusManager()
        #if busmanager invalid
        try:
            self.numCams = self.bus.getNumOfCameras()
            print('Found cameras!')
        except:
            self.numCams = 0
        #set up camera if available
        if self.numCams == 0:
            print("No suitable USB cameras found...")
        else:
            self.RGBTopCamera = PyCapture2.Camera()
            # print(self.bus)
            RGBTopUid = self.bus.getCameraFromIndex(0)
            #RGBTopUid = self.bus.getCameraFromSerialNumber(15435621)
            self.RGBTopCamera.connect(RGBTopUid)
            self.RGBTopCamera.startCapture()
	    self.camConf = self.RGBTopCamera.getConfiguration()
	    self.RGBTopCamera.setProperty(type = PyCapture2.PROPERTY_TYPE.AUTO_EXPOSURE, autoManualMode = True)
	    self.RGBTopCamera.setProperty(type = PyCapture2.PROPERTY_TYPE.SHARPNESS, autoManualMode = True)
	    self.RGBTopCamera.setProperty(type = PyCapture2.PROPERTY_TYPE.SHUTTER, autoManualMode = False)
	    self.RGBTopCamera.setProperty(type = PyCapture2.PROPERTY_TYPE.GAIN, autoManualMode = False)
	    self.RGBTopCamera.setProperty(type = PyCapture2.PROPERTY_TYPE.FRAME_RATE, autoManualMode = True)
	    self.RGBTopCamera.setProperty(type = PyCapture2.PROPERTY_TYPE.AUTO_EXPOSURE, onOff = True)
	    self.RGBTopCamera.setProperty(type = PyCapture2.PROPERTY_TYPE.FRAME_RATE, onOff = True)
	    self.RGBTopCamera.setProperty(type = PyCapture2.PROPERTY_TYPE.GAMMA, onOff = True)
	    self.RGBTopCamera.setProperty(type = PyCapture2.PROPERTY_TYPE.SHARPNESS, onOff = False)

	    SHUTTER, GAIN = 500, 100
	    self.RGBTopCamera.setProperty(type = PyCapture2.PROPERTY_TYPE.SHUTTER, absValue = SHUTTER/10)
	    self.RGBTopCamera.setProperty(type = PyCapture2.PROPERTY_TYPE.GAIN, absValue = GAIN/20)
	    # Configure trigger mode
#	    triggerMode = c.getTriggerMode()
#	    triggerMode.onOff = True
#	    triggerMode.mode = 0
#	    triggerMode.parameter = 0
#	    triggerMode.source = 0    #Not sure???
#	    self.RGBTopCamera.setTriggerMode(triggerMode)
#	    self.RGBTopCamera.setConfiguration(grabTimeout = 5000)  

    '''
    RUN
    '''
    def run(self):
        #do 10 for now
        for i in range(1000):
            #time between images
            #time.sleep(.5)
            print("Capturing....")
            start = timer()
            # check if camera is connected
            if self.RGBTopCamera is not None:
                #if so, get buffer
                image = self.RGBTopCamera.retrieveBuffer()
                #add buffer to numpy array
                cv_image = np.array(image.getData(), dtype="uint8").reshape((image.getRows(), image.getCols()));
                #convert to colour
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BAYER_BG2BGR)
                # print(cv_image.shape)
            #else, use the static image
            ## TODO: Replace with folder images
            else:
                cv_image = self.cv_image

            #get an object lock
            self.lock.acquire()
            #add the image to the parent punnet
            self.punnet.RGBTopImage = cv_image
            #set display flag
            self.punnet.punnetNeedsDisplaying = True
            end = timer()
            duration = end - start
            print(duration)
            self.lock.release()

        #if the camera is connected,
        # MUST disconnect to stop
        # 'Invalid Bus Manager' error
        if self.RGBTopCamera is not None:
            print('Disconnecting camera....')
            self.RGBTopCamera.stopCapture()
            self.RGBTopCamera.disconnect()
        self.bus = None
