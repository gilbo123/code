import threading
import numpy as np
import cv2
import time
import PyCapture2
from timeit import default_timer as timer

threadLock = threading.Lock()

class CamWorker(threading.Thread):
    # Our workers constructor, note the super() method which is vital if we want this
    # to function properly
    def __init__(self, lock, punnet):
        super(CamWorker, self).__init__()
        self.blank_img = np.zeros((600, 500, 3),dtype=np.uint8)
        self.cv_image = cv2.imread('/home/gilbert/Documents/code/openCV/transfer_learn/berries/pass/20180427_220416_0.jpg')
        self.lock = lock
        self.punnet = punnet
        #camera set up - RGBTop
        # self.bus = PyCapture2.BusManager()
        # self.RGBTopCamera = PyCapture2.Camera()
        # print(self.bus)
        # RGBTopUid = self.bus.getCameraFromIndex(0)
        # #RGBTopUid = self.bus.getCameraFromSerialNumber(15435621)
        # self.RGBTopCamera.connect(RGBTopUid)
        # self.RGBTopCamera.startCapture()


    def run(self):
        for i in range(10):
            time.sleep(1)
            print("Capturing....")
            start = timer()
            # image = self.RGBTopCamera.retrieveBuffer()
            # cv_image = np.array(image.getData(), dtype="uint8").reshape((image.getRows(), image.getCols()));

            threadLock.acquire()
            self.punnet.RGBTopImage = self.cv_image
            self.punnet.punnetNeedsDisplaying = True
            end = timer()
            duration = end - start
            print(duration)
            threadLock.release()


            # threadLock.acquire()
            # self.punnet.RGBTopImage = self.cv_image
            # print(i)
            # threadLock.release()
            # time.sleep(.5)
            #
            # threadLock.acquire()
            # self.punnet.RGBTopImage = self.blank_img
            # threadLock.release()
            # time.sleep(.5)
        self.bus = None
