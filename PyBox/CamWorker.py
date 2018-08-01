import threading
from sys import exit
import numpy as np
import cv2
import time
import PySpin
from timeit import default_timer as timer

threadLock = threading.Lock()

class TRIG:
    TRIGGER_ON = True
    TRIGGER_OFF = False


class CamWorker(threading.Thread):
    # Our workers constructor, note the super() 
    # method which is vital if we want this
    # to function properly
    def __init__(self, lock, sys, punnet, camName):
        super(CamWorker, self).__init__()
        self.blank_img = np.zeros((600, 500, 3),dtype=np.uint8)
        self.cv_image = cv2.imread('/home/gilbert/Documents/code/'+
				'openCV/transfer_learn/berries/pass/'+
				'20180427_220416_0.jpg')
        self.lock = lock
        self.punnet = punnet
	self.camName = camName
	self.sys = sys
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



     
    
    #def setParams(self, cam, triggerMode):
#	try:
	    

#	except PySpin.SpinnakerException as ex:
 #           print 'Error: %s' % ex
  #          return False	




    def initCam(self):
	try:
            # Retrieve singleton reference to system object
            self.sys = PySpin.System.GetInstance()
            # Retrieve list of cameras from the system
            self.cam_list = self.sys.GetCameras()
            # assign cam object to self.cam (one camera run)
            self.cam = self.cam_list.GetByIndex(self.camIndex)
            #print(dir(self.cam))
	    
	    #set cam parameters
	    #setParams(self.cam, TRIG.TRIGGER_ON)
	      


	    # initialize cam object
            self.cam.Init()
            self.is_connected = True


	    
            # get nodemap
            nodemap = self.cam.GetNodeMap()
            node_trigger_mode = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerMode'))
            if not PySpin.IsAvailable(node_trigger_mode) or not PySpin.IsReadable(node_trigger_mode):
                print 'Unable to disable trigger mode (node retrieval). Aborting...'
                return False

	    #should be a check before setting every param??
	    
	    #set hardware trigger
	    node_trigger_source = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerSource'))
	    node_trigger_source_hardware = node_trigger_source.GetEntryByName('Line0')
	    node_trigger_source.SetIntValue(node_trigger_source_hardware.GetValue())
	    #turn on trigger
	    node_trigger_mode_on = node_trigger_mode.GetEntryByName('Off')
	    node_trigger_mode.SetIntValue(node_trigger_mode_on.GetValue())


	    #  Begin acquiring images
            self.cam.BeginAcquisition()
            print('Connected.....')
	    return self.is_connected
	    

        except PySpin.SpinnakerException as ex:
	    print('Error initializing - {}'.format(ex))
	    #camera not connected,
	    #index = -1
	    self.camIndex = -1 
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
	    if self.camIndex > -1 and self.is_connected:
		#  Retrieve next received image
                raw_image = self.cam.GetNextImage()
		image = raw_image.GetNDArray()
		image = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2BGR)	
		filename = self.camName + '-%d.jpg' % i

	    else:
		#get image from file
		image = self.cv_image
		time.sleep(.02)

            #Save image
	    #cv2.imwrite(filename, image)

	    #get an object lock
            self.lock.acquire()

            #add the image to the parent punnet 
	    if self.camName == 'RGBTop':
		print('RGBTop image')
	    	self.punnet.RGBTopImage = image	
	    elif self.camName == 'IRTop':
		print('IRTop image')
		self.punnet.IRTopImage = image
                self.punnet.punnetNeedsDisplaying = True
	    elif self.camName == 'RGBBtm':
		print('RGBBtm image')
		self.punnet.RGBBtmImage = image
	    elif self.camName == 'IRBtm':
		print('IRBtm image')
		self.punnet.IRBtmImage = image
            
	    #set display flag
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
	    # Deinitialize camera 
            self.cam.EndAcquisition()
	    self.cam.DeInit()
	    del self.cam

