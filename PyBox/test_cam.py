import PyCapture2
import numpy as np
import cv2
from matplotlib import pyplot as plt

winSize = 300
winCorner = 5
borderSpace = 80

windows = {'RGBTop': [winCorner, winCorner], 'IRTop': [winCorner, winCorner + winSize + borderSpace ],
            'RGBBtm': [winCorner + winSize + borderSpace, winCorner],
            'IRBtm':[winCorner + winSize + borderSpace, winCorner + winSize + borderSpace]}
# HWArr = {{200, 200}, {400, 200}, {200, 400}, {400, 400}}

def openWindows():
    i=0
    for window, pos in windows.items():
        # print(window)
        blank_img = np.zeros((600, 500, 3),dtype=np.uint8)
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, winSize, winSize)
        cv2.imshow(window, blank_img)
        h = pos[0]
        w = pos[1]
        cv2.moveWindow(window, h, w)  # Move it to (40,30)
        i+=1

if __name__=="__main__":
    openWindows()
    # try:
        # bus = PyCapture2.BusManager()
        # print(bus)
        # numCams = bus.getNumOfCameras()
        # uid = bus.getCameraFromIndex(0)
        # camera = PyCapture2.Camera()
        #
        # camera.connect(uid)
        # #print(camera.getPropertyInfo())
        # camera.startCapture()

    for i in range(10):
        cv_image = cv2.imread('/home/gilbert/Documents/code/openCV/transfer_learn/berries/pass/20180427_220416_0.jpg')
        # image = camera.retrieveBuffer()
        # cv_image = np.array(image.getData(), dtype="uint8").reshape((image.getRows(), image.getCols()) );
        for window, pos in windows.items():
            cv2.imshow(window, cv_image)

        cv2.waitKey()

    # except:
    #     print('Exiting....')
    #     bus = None
    cv2.destroyAllWindows()
    bus = None
