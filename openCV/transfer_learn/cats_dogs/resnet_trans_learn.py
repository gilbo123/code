# import the necessary packages
from keras.applications import ResNet50
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.preprocessing.image import apply_transform
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

from imutils import paths
import numpy as np
import progressbar
import random
import os


#from this website
#https://blogs.technet.microsoft.com/machinelearning/2018/02/22/22-minutes-to-2nd-place-in-a-kaggle-competition-with-deep-learning-azure/


# since we are not using command line arguments (like we typically
# would inside Deep Learning for Computer Vision with Python, let's
# "pretend" we are by using an `args` dictionary -- this will enable
# us to easily reuse and swap out code depending if we are using the
# command line or Jupyter Notebook
args = {
    "dataset": "../train2",
    "batch_size": 32,
}

# store the batch size in a convenience variable
bs = args["batch_size"]

# grab the list of images in the Kaggle Dogs vs. Cats download and
# shuffle them to allow for easy training and testing splits via
# array slicing during training time
imagePaths = list(paths.list_images(args["dataset"]))
random.shuffle(imagePaths)
print(len(imagePaths))

# extract the class labels from the image paths then encode the
# labels
labels = [p.split(os.path.sep)[-1].split(".")[0] for p in imagePaths]
le = LabelEncoder()
labels = le.fit_transform(labels)

# load the ResNet50 network (i.e., the network we'll be using for
# feature extraction)
model = ResNet50(weights="imagenet", include_top=False)#classes=1000

# initialize the progress bar
widgets = ["Extracting Features: ", progressbar.Percentage(), " ",
    progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(imagePaths), widgets=widgets).start()

# initialize our data matrix (where we will store our extracted
# features)
data = None

# loop over the images in batches
for i in np.arange(0, len(imagePaths), bs):
    # extract the batch of images and labels, then initialize the
    # list of actual images that will be passed through the network
    # for feature extraction
    batchPaths = imagePaths[i:i + bs]
    batchLabels = labels[i:i + bs]
    batchImages = []

    # loop over the images and labels in the current batch
    for (j, imagePath) in enumerate(batchPaths):
        # load the input image using the Keras helper utility
        # while ensuring the image is resized to 224x224 pixels
        image = load_img(imagePath, target_size=(224, 224))
        image = img_to_array(image)

        # preprocess the image by (1) expanding the dimensions and
        # (2) subtracting the mean RGB pixel intensity from the
        # ImageNet dataset
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)

        # add the image to the batch
        batchImages.append(image)
        

    # pass the images through the network and use the outputs as
    # our actual features
    batchImages = np.vstack(batchImages)
    features = model.predict(batchImages, batch_size=bs)

    # reshape the features so that each image is represented by
    # a flattened feature vector of the `MaxPooling2D` outputs
    features = features.reshape((features.shape[0], 2048))

    # if our data matrix is None, initialize it
    if data is None:
        data = features

    # otherwise, stack the data and features together
    else:
        data = np.vstack([data, features])

    # update the progress bar
    pbar.update(i)

# finish up the progress bar
pbar.finish()


# show the data matrix shape and amount of memory it consumes
print(data.shape)
print(data.nbytes)

# determine the index of the training and testing split (75% for
# training and 25% for testing)
i = int(data.shape[0] * 0.75)

# define the set of parameters that we want to tune then start a
# grid search where we evaluate our model for each value of C
print("[INFO] tuning hyperparameters...")
params = {"C": [0.0001, 0.001, 0.01, 0.1, 1.0]}
clf = GridSearchCV(LogisticRegression(), params, cv=3, n_jobs=-1)
clf.fit(data[:i], labels[:i])
print("[INFO] best hyperparameters: {}".format(clf.best_params_))

# generate a classification report for the model
print("[INFO] evaluating...")
preds = clf.predict(data[i:])
print(classification_report(labels[i:], preds, target_names=le.classes_))

# compute the raw accuracy with extra precision
acc = accuracy_score(labels[i:], preds)
print("[INFO] score: {}".format(acc))

#save model
joblib.dump(clf, 'cats_dogs.pkl')
