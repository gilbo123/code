import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

path = 'transfer_learn/berries/pass/'

def getImagePaths(path):
    #set counter
    imagePaths = []
    #print(name)
    files = os.walk(path).next()[2]
    if (len(files) > 0):
        for file in files:
            #print(file)
            imagePath = os.path.join(path, file)
            imagePaths.append(imagePath)

    return imagePaths

paths = getImagePaths(path)

# loop over the images in batches
for i in np.arange(0, 296):#len(paths)):
  img = plt.imread(paths[i], 0)

  horizontal_img = cv2.flip(img, 0)
  vertical_img = cv2.flip( img, 1 )
  both_img = cv2.flip( img, -1 )

  #add a rotated image
  # 90 degrees
  # M1 = cv2.getRotationMatrix2D(center, angle90, scale)
  # rotated_90 = cv2.warpAffine(img, M1, (h, w))
  # p_rotated_90 = fun.preProcessImage(rotated_90)
  # #270 degrees
  # M2 = cv2.getRotationMatrix2D(center, angle270, scale)
  # rotated_270 = cv2.warpAffine(img, M2, (h, w))
  # p_rotated_270 = fun.preProcessImage(rotated_270)

  RGB1 = cv2.cvtColor(horizontal_img, cv2.COLOR_BGR2RGB)
  cv2.imwrite( path + "hor_transformed_" + str(i) + ".jpg", RGB1 );

  RGB2 = cv2.cvtColor(vertical_img, cv2.COLOR_BGR2RGB)
  cv2.imwrite( path + "ver_transformed_" + str(i) + ".jpg", RGB2 );

  RGB3 = cv2.cvtColor(both_img, cv2.COLOR_BGR2RGB)
  cv2.imwrite( path + "h_v_transformed_" + str(i) + ".jpg", RGB3 );
