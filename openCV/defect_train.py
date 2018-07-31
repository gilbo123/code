from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras import optimizers

# dimensions of our images.
img_width, img_height = 500,600

#put classes in folders
train_data_dir = 'transfer_learn/berries/'
validation_data_dir = 'TEST\\'
nb_validation_samples = 43
nb_train_samples = 362
epochs = 100
batch_size = 20

if K.image_data_format() == 'channels_first':
    input_shape = (1, img_width, img_height)
else:
    input_shape = (img_width, img_height, 1)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dropout(0.25))
model.add(Dense(1,activation='sigmoid'))
#model.add(Dense(1))
#model.add(Activation('sigmoid'))

opt = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(loss='binary_crossentropy',
#              optimizer='rmsprop',
              optimizer=opt,
              metrics=['accuracy'])

## this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
#    shear_range=0.2,
#    zoom_range=0.2,
#    horizontal_flip=True,
#    vertical_flip=True
    )
#train_datagen = ImageDataGenerator()
validation_datagen = ImageDataGenerator(
    rescale=1. / 255,
#    shear_range=0.2,
#    zoom_range=0.2,
#    horizontal_flip=True,
#    vertical_flip=True
    )

# this is the augmentation configuration we will use for testing:
# only rescaling

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    color_mode='grayscale',
#    color_mode='rgb',
    shuffle = True)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
#    batch_size=batch_size,
    class_mode='binary',
    color_mode='grayscale',
#    color_mode='rgb',
    shuffle = True)
#%%
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps = 43,
    verbose=1)
#%%
from keras.utils.vis_utils import plot_model
plot_model(model, to_file="model.png")
