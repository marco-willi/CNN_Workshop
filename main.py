# import modules
import keras
from keras.datasets import mnist
from keras.preprocessing.image import array_to_img, load_img
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from shutil import copyfile, rmtree


# Path to image data
path = 'D:/Studium_GD/Zooniverse/Data/cnn_workshop/PetImages/'
# prepare paths
group_name = 'group1/'
os.mkdir(path + group_name)
os.mkdir(path + group_name + 'train')
os.mkdir(path + group_name + 'test')


# Look at some images
def look_at_imgs(path, n=5):
    files = os.listdir(path)
    # choose random images
    random.seed(23)
    random_ids = random.sample(population=range(0, len(files)), k=n)
    random_imgs = [files[x] for x in random_ids]

    # display
    for i in range(0, len(random_ids)):
        plt.imshow(load_img(path + random_imgs[i]))
        plt.show()
        print(path + random_imgs[i])

look_at_imgs(path + 'Cat/')
look_at_imgs(path + 'Dog/')


# Generate  Training / Test / Validation Split
from sklearn.model_selection import train_test_split

# get all files and labels
files = [path + 'Cat/' + x for x in os.listdir(path + 'Cat')]
labels = ['Cat' for x in range(0, len(files))]
dog_files = [path + 'Dog/' + x for x in os.listdir(path + 'Dog')]
labels_dog = ['Dog' for x in range(0, len(dog_files))]
files.extend(dog_files)
labels.extend(labels_dog)

# generate splits
id_train, id_test = train_test_split(files,
                                     train_size=2000,
                                     test_size=1000,
                                     stratify=labels,
                                     random_state=234)

# create directories
def create_new(path):
    # create new directory
    if os.path.isdir(path):
        rmtree(path)
    os.mkdir(path)

create_new(path + group_name + 'train/' + 'Dog/')
create_new(path + group_name + 'test/' + 'Dog/')
create_new(path + group_name + 'train/' + 'Cat/')
create_new(path + group_name + 'test/' + 'Cat/')

# copy training / test files to dedicated dirs
def copy_files(path, files):
    for f in files:
        label = f.split('/')[-2]
        fname = f.split('/')[-1]
        copyfile(f, path + label + '/' + fname)

copy_files(path=path + group_name + 'train/', files=id_train)
copy_files(path=path + group_name + 'test/', files=id_test)


# Create Data Generator
# pre-processing of images
from keras.preprocessing.image import ImageDataGenerator

# parameters
batch_size = 32

# generator for training data
datagen_train = ImageDataGenerator(
    rescale=1./255,
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

# generator for test data (no data augmentation!)
datagen_test = ImageDataGenerator(
    rescale=1./255,
    featurewise_center=True,
    featurewise_std_normalization=True)

# generator to get raw images to estimate transformations
datagen_raw = ImageDataGenerator(
    rescale=1./255
    )

raw_gen = datagen_raw.flow_from_directory(
    path + group_name + 'train/',
    target_size=(150, 150),
    batch_size=1000,
    class_mode='binary',
    seed=123)

X_raw = raw_gen.next()

# fit data generators on raw data
datagen_train.fit(X_raw[0])
datagen_test.fit(X_raw[0])

# fetch data from directory in specified batch sizes
train_generator = datagen_train.flow_from_directory(
    path + group_name + 'train/',
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='binary',
    seed=123)

test_generator = datagen_test.flow_from_directory(
    path + group_name + 'test/',
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='binary',
    seed=123)

# dummy generator to look at data
dummy_gen = datagen_train.flow_from_directory(
    path + group_name + 'train/',
    target_size=(150, 150),
    batch_size=5,
    class_mode='binary',
    seed=123)

# lets take a  look what the data generator actually does
data_batch = dummy_gen.next()
for i in range(0, len(data_batch[1])):
    print("Label: %s" % data_batch[1][i])
    plt.imshow(data_batch[0][i, :])
    plt.show()


# Define a model architecture
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.models import Model

# sequential model (layer based)
model = Sequential()
# Convolutional layer over 2 dimensions
# 32 filters, each with a size of 3x3 pixels
model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
# activation function
model.add(Activation('relu'))
# Aggregate data using max pooling (reduce size of feature maps)
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# this converts our 3D feature maps to 1D feature vectors
model.add(Flatten())
# fully connected layer with 64 output values
model.add(Dense(64))
# applies static function on output
model.add(Activation('relu'))
# randomly set 50% of all outputs to zero to prevent overfitting
model.add(Dropout(0.5))
# fully connected layer with 1 output value
model.add(Dense(1))
# sigmoid transformation (logistic regression) to obtain class probability
model.add(Activation(activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


# train the model
model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.n // batch_size,
    epochs=5,
    workers=2,
    validation_data=test_generator,
    validation_steps=test_generator.n // batch_size)


# Load a pre-trained model
from keras.applications.inception_v3 import InceptionV3

# load Googles inception model with imagenet weights
base_model = InceptionV3(
    include_top=False,
    weights='imagenet',
    input_shape=(150, 150, 3))

# take a look at its architecture
base_model.summary()

# set layers to non-trainable
for layer in base_model.layers:
    layer.trainable = False

# add a global spatial average pooling layer
top_model = base_model.output
top_model = GlobalAveragePooling2D()(top_model)
top_model = Dense(1, activation='relu')(top_model)
top_model = Activation(activation='sigmoid')(top_model)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=top_model)


model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.summary()

# feed data to the training pre-processing
# WARNING: make sure the pre-processing is the same as used to train the
# pre-trained model
model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.n // batch_size,
    epochs=5,
    workers=2,
    validation_data=test_generator,
    validation_steps=test_generator.n // batch_size)
