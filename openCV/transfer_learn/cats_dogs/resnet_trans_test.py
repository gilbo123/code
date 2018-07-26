# import the necessary packages
from keras.applications import ResNet50
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

from timeit import default_timer as timer
from matplotlib import pyplot as plt
from imutils import paths
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
    "dataset": "../berries",
    "batch_size": 32,
}

#get paths
imagePaths = list(paths.list_images(args["dataset"]))
random.shuffle(imagePaths)
print(len(imagePaths))

# load the ResNet50 network (i.e., the network we'll be using for
# feature extraction)
model = ResNet50(weights="imagenet", include_top=False)

#load the model trained for cats and dogs
clf = joblib.load('cats_dogs.pkl')

data = None
batchImages = []
# loop over the images and labels in the current batch
for (j, imagePath) in enumerate(imagePaths):
    #get image from path
    img = cv2.imread(imagePath)
    #show image
    cv2.imshow("Image", img)

    #timer to time NN
    start = timer()

    # load the input image using the Keras helper utility
    # while ensuring the image is resized to 224x224 pixels
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)

    # preprocess the image by (1) expanding the dimensions and
    # (2) subtracting the mean RGB pixel intensity from the
    # ImageNet dataset
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)


    features = model.predict(image)

    # reshape the features so that each image is represented by
    # a flattened feature vector of the `MaxPooling2D` outputs
    features = features.reshape((features.shape[0], 2048))

    # if our data matrix is None, initialize it
    if data is None:
        data = features

    # otherwise, stack the data and features together
    else:
        data = np.vstack([data, features])

    #Predict result
    res = clf.predict(data)
    prob = clf.predict_proba(data)

    #for some reason can return out of order
    if prob[j][0] > prob[j][1]:
        #first var
        probability = prob[j][0]
    else:
        probability = prob[j][1]

    #end timer
    end = timer()
    time = end-start

    if (res[j] == 1):
        name = 'Dog'
    else:
        name = 'Cat'

    #stats
    print "Image is a: ", name
    print "Probability: ", probability
    print "In: ", time, "seconds\r\n"
    print features[0]

    #feature signature
    # Plot the histogram
    plt.figure()
    plt.title("Feature Signature")
    plt.xlabel("Vector Indices")
    plt.ylabel("Strength")
    plt.plot(features[0])
    plt.xlim([0, 2047])
    plt.show()


    #wait til keypress
    k = cv2.waitKey(0)
    if k==27:
        cv2.destroyAllWindows()
    	break
