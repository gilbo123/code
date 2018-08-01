import threading
from sys import exit
import numpy as np
import cv2
import time
import PySpin
from timeit import default_timer as timer

threadLock = threading.Lock()


class CamWorker(threading.Thread):
    # Our workers constructor, note the super() 
    # method which is vital if we want this
    # to function properly
    def __init__(self, lock, punnet, camName):
        super(CamWorker, self).__init__()
        self.blank_img = np.zeros((600, 500, 3),dtype=np.uint8)
        self.cv_image = cv2.imread('/home/gilbert/Documents/code/'+
				'openCV/transfer_learn/berries/pass/'+
				'20180427_220416_0.jpg')
        self.lock = lock
        self.punnet = punnet
	self.camName = camName
	#check cam name 
	if self.camName == 'RGBTop':
	    self.camIndex = 0
	elif self.camName == 'IRTop':
	    self.camIndex = 1
	elif self.camName == 'RGBBtm':
	    self.camIndex = 2
	elif self.camName == 'IRBtm':
	    self.camIndex = 3
	else:
	    print('Camera Name not correct')
	    exit()
        self.cam = None
	self.sys = None
	self.cam_list = None
	self.is_connected = False



    def initCam(self):
	try:
            # Retrieve singleton reference to system object
            self.sys = PySpin.System.GetInstance()
            # Retrieve list of cameras from the system
            self.cam_list = self.sys.GetCameras()
            # assign cam object to self.cam (one camera run)
            self.cam = self.cam_list.GetByIndex(self.camIndex)
            print(dir(self.cam))
	    # initialize cam object
            self.cam.Init()
            self.is_connected = True

	    #  Begin acquiring images
            self.cam.BeginAcquisition()
            print('Connected.....')
	    return self.is_connected
	    

        except PySpin.SpinnakerException as ex:
	    print('Error initializing - {}'.format(ex))
            return False




    '''
    RUN
    '''
    def run(self):
        #do 10 for now
        for i in range(100):
            #time between images
            #time.sleep(.5)
            print("Capturing....")
            start = timer()
            
	    # check if camera is connected
	    if self.cam_list > 0 and self.is_connected:
		#  Retrieve next received image
                raw_image = self.cam.GetNextImage()
		image = raw_image.GetNDArray()
		image = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2BGR)	
		filename = self.camName + '-%d.jpg' % i
                #image.Release()

	    else:
		#get image from file
		image = self.cv_image

            #Save image
	    #cv2.imwrite(filename, image)

	    #get an object lock
            self.lock.acquire()

            #add the image to the parent punnet 
	    if self.camName == 'RGBTop':
	    	self.punnet.RGBTopImage = image
	    elif self.camName == 'IRTop':
		self.punnet.IRTopImage = image
	    elif self.camName == 'RGBBtm':
		self.punnet.RGBBtmImage = image
	    elif self.camName == 'IRBtm':
		self.punnet.IRBtmImage = image
            
	    #set display flag
            self.punnet.punnetNeedsDisplaying = True
            end = timer()
            duration = end - start
            print(duration)
            
	    #release lock
	    self.lock.release()

        #if the camera is connected,
	# End acquisition
	if self.is_connected:
	    #release image 
	    raw_image.Release()
	    time.sleep(2)
	    # Deinitialize camera 
            self.cam.EndAcquisition()
	    self.cam.DeInit()
	    del self.cam

