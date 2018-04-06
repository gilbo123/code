# import the necessary packages
from keras.applications import ResNet50
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from scipy.spatial import distance
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

from timeit import default_timer as timer
from matplotlib import pyplot as plt
from imutils import paths
from Functions import Functions
import tables
import numpy as np
import progressbar
import random
import os
import cv2

#from this website
#https://blogs.technet.microsoft.com/machinelearning/2018/02/22/22-minutes-to-2nd-place-in-a-kaggle-competition-with-deep-learning-azure/


# since we are not using command line arguments (like we typically
# would inside Deep Learning for Computer Vision with Python, let's
# "pretend" we are by using an `args` dictionary -- this will enable
# us to easily reuse and swap out code depending if we are using the
# command line or Jupyter Notebook
args = {
    "dataset": "../berries/test/",
}

#batch size
bs = 16

#functions class
fun = Functions()

#image to look for
imageType = 'RGBTop'

#get numer of images
imagePaths = fun.getImagePaths(args, imageType)

#Get the data saved to files
# u_h5 = tables.open_file("under.h5", "r")
ok_h5 = tables.open_file("ok.h5", "r")
# under_data = u_h5.root
ok_data = ok_h5.root
# print(ok_h5)

# load the ResNet50 network (i.e., the network we'll be using for
# feature extraction)
model = ResNet50(weights="imagenet", include_top=False)

for path in imagePaths:
    #timer to time method
    start = timer()

    # load the input image using the Keras helper utility
    image = fun.preProcessImage(path)

    #get features from network
    features = model.predict(image)

    # reshape the features so that each image is represented by
    # a flattened feature vector of the `MaxPooling2D` outputs
    features = features.reshape((features.shape[0], 2048))#8192

    #numpy format 32-bit
    features = np.array(features, np.float32)

    #set the name first
    name = 'OK'
    #Predict result
    #comparrison to stored arrays
    distances = []
    for leaf in ok_data._f_walknodes('Leaf'):
        # Get euclidian distances between arrays
        dists = distance.cdist((np.asarray(features)).reshape(1, -1), (np.asarray(leaf).reshape(1, -1)))
        print dists
        distances.append(dists)


    #metrics
    print('Min dist: {}'.format(np.min(distances)))
    print('Max dist: {}'.format(np.max(distances)))
    print('Range: {}'.format(np.max(distances) - np.min(distances)))
    print('Mean dist: {}'.format(np.mean(distances)))
    print('Std Dev (pop): {}'.format(np.std(distances)))
    print('Std Dev (sam): {}'.format(np.std(distances, ddof=1)))

    if np.max(distances) > 20:
        name = 'Underripe'

    #end timer
    end = timer()
    time = end-start

    #stats
    print "Image is : ", name
    #print "Probablility: ", score[0]
    print "In: ", time, "seconds\r\n"

    #get image
    img = plt.imread(path, 0)
    plt.imshow(img)
    plt.show()
    cv2.waitKey(0)


# u_h5.close()
ok_h5.close()
