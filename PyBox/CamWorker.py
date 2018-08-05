import threading
from sys import exit
import numpy as np
import cv2
import time
import PySpin
from timeit import default_timer as timer



'''
Worker class - feeds the main thread with images from 
	various cameras. 

Inherits: Thread
Args: 1. lock - threadlock from main
      2. sys - PySpin system instance
      3. cam_list - list of all cameras
      4. punnet - global punnet object
      5. camName - camera name to determine image type

'''
class CamWorker(threading.Thread):
    # Our workers constructor, note the super() 
    # method which is vital if we want this
    # to function properly
    def __init__(self, lock, sys, cam_list, punnet, camName):
        super(CamWorker, self).__init__()
        self.blank_img = np.zeros((600, 500, 3),dtype=np.uint8)
        self.cv_image = cv2.imread('/home/gilbert/Documents/code/'+
				'openCV/transfer_learn/berries/pass/'+
				'20180427_220416_0.jpg')
        self.stopSignal = False
        self.lock = lock
        self.punnet = punnet
	self.camName = camName
	self.sys = sys
	self.cam_list = cam_list
	#check cam name 
	if self.camName == 'RGBTop':
	    self.camIndex = 1
	elif self.camName == 'IRTop':
	    self.camIndex = 0
	elif self.camName == 'RGBBtm':
	    self.camIndex = 2
	elif self.camName == 'IRBtm':
	    self.camIndex = 3
	else:
	    print('Camera Name not correct')
	    exit()
        self.cam = None
	self.is_connected = False


    
    #Finds the camera by name, sets relevant params
    # and finally, initialises the camera
    def initCam(self):
	try:
	    #set the camera for this instance
            self.cam = self.cam_list.GetBySerial('16097126')
	    
	    # initialize cam object
            self.cam.Init()
            self.is_connected = True


            #only set params of colour cam
	    if self.camName == 'RGBTop' or self.camName == 'RGBBtm':
		    '''
		    Get NodeMap for parameter setting
		    '''
		    # get nodemap
		    nodemap = self.cam.GetNodeMap()
		    node_trigger_mode = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerMode'))
		    #Check the IsAvailable() and IsReadable() once. The examples
		    #check these for each step of the parameter setting, not sure
		    #if needed?  
		    if not PySpin.IsAvailable(node_trigger_mode) or not PySpin.IsReadable(node_trigger_mode):
		    	print 'Unable to disable trigger mode (node retrieval). Aborting...'
		    	exit()
		    
		    '''
		    Set TRIGGER
		    '''
		    #set hardware trigger
		    node_trigger_source = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerSource'))
		    node_trigger_source_hardware = node_trigger_source.GetEntryByName('Line0')
		    node_trigger_source.SetIntValue(node_trigger_source_hardware.GetValue())
		    #turn on trigger
		    node_trigger_mode_on = node_trigger_mode.GetEntryByName('Off')
		    node_trigger_mode.SetIntValue(node_trigger_mode_on.GetValue())


		    '''
		    Set EXPOSURE
		    '''
		    #set exposure mode
		    node_exp = PySpin.CEnumerationPtr(nodemap.GetNode('ExposureAuto'))
		    node_exp_mode = node_exp.GetEntryByName('Off')
		    node_exp.SetIntValue(node_exp_mode.GetValue())
		    node_exp_timed = PySpin.CEnumerationPtr(nodemap.GetNode('ExposureMode'))
		    node_exp_timed_val = node_exp_timed.GetEntryByName('Timed')
		    node_exp_timed.SetIntValue(node_exp_timed_val.GetValue())
		    #set exposure time
		    node_var = PySpin.CFloatPtr(nodemap.GetNode('ExposureTime'))
		    node_var.SetValue(2000)#23872 - max


		    '''
		    Set pgr_EXPOSURE_COMPENSATION
		    '''
		    #set point grey automatic exposure compensation - off
		    node_pgr_comp = PySpin.CEnumerationPtr(nodemap.GetNode('pgrExposureCompensationAuto'))
		    node_pgr_comp_mode = node_pgr_comp.GetEntryByName('Off')
		    node_pgr_comp.SetIntValue(node_pgr_comp_mode.GetValue())


		    '''
		    Set GAIN
		    '''
		    #set point grey automatic gain - off
		    node_gain = PySpin.CEnumerationPtr(nodemap.GetNode('GainAuto'))
		    node_gain_mode = node_gain.GetEntryByName('Off')
		    node_gain.SetIntValue(node_gain_mode.GetValue())
		    #set exposure time
		    node_gain_val = PySpin.CFloatPtr(nodemap.GetNode('Gain'))
		    node_gain_val.SetValue(10)#30 - max



		    '''
		    Set WHITE_BALANCE
		    '''
		    #set exposure mode
	#           node_exp = PySpin.CEnumerationPtr(nodemap.GetNode('BalanceWhiteAuto'))
	#           node_exp_mode = node_exp.GetEntryByName('Off')
	#           node_exp.SetIntValue(node_exp_mode.GetValue())
	#           node_exp_timed = PySpin.CEnumerationPtr(nodemap.GetNode('BalanceRatioSelector'))
	#           node_exp_timed_val = node_exp_timed.GetEntryByName('Red')
	#           node_exp_timed.SetIntValue(node_exp_timed_val.GetValue())
		    #set exposure time
	#           node_var = PySpin.CFloatPtr(nodemap.GetNode('BalanceRatio'))
	#           node_var.SetValue(1.22)#4 - max



	    #  Begin acquiring images
            self.cam.BeginAcquisition()
            print('Connected.....')
	    return self.is_connected
	    

        except PySpin.SpinnakerException as ex:
	    print('Error initializing {0} - {1}'.format(self.camName, ex))
	    #camera not connected,
	    #index = -1
	    self.camIndex = -1 
            return False



    def stop(self):
        self.stopSignal = True      
        #release image 
        #raw_image.Release()
        # Deinitialize camera 
        self.cam.EndAcquisition()
        self.cam.DeInit()
        del self.cam
        print('DONE DONE DONE!')




    '''
    RUN
    '''
    def run(self):
        while(not self.stopSignal):
	    #time between images
	    print("Capturing....")
	    start = timer()
	    
	    #image placeholder
	    image = None
	    # check if camera is connected
	    if self.camIndex > -1 and self.is_connected:
		#  Retrieve next received image
		try:
		    raw_image = self.cam.GetNextImage() #Timeout in ms
		    image = raw_image.GetNDArray()
		    image = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2BGR)
                    print('Image acquired!')
		    #filename = self.camName + '-%d.jpg' % i
		except:
		    print('Camera Timeout!')
	    else:
		#get image from file
		image = self.cv_image
		time.sleep(.05)

	    #Save image
	    #cv2.imwrite(filename, image)

	    #get an object lock
	    self.lock.acquire()

	    #add the image to the parent punnet 
	    if self.camName == 'RGBTop':
		print('RGBTop image')
		self.punnet.punnetNeedsDisplaying = True
		self.punnet.RGBTopImage = image	
	    elif self.camName == 'IRTop':
		print('IRTop image')
		self.punnet.IRTopImage = image
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
	#if self.is_connected:
	    #release image 
	#    raw_image.Release()
	    # Deinitialize camera 
	#    self.cam.EndAcquisition()
	#    self.cam.DeInit()
	#    del self.cam
        #    print('DONE DONE DONE!')


