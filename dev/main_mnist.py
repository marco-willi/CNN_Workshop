# import modules
from keras.datasets import mnist
from keras.preprocessing.image import array_to_img
import matplotlib.pyplot as plt
import numpy as np
import random

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)

# inspect data
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# choose random images
random.seed(23)
random_ids = random.sample(population=range(0, x_train.shape[0]),k=5)
random_imgs = x_train[random_ids, :]
random_labels = y_train[random_ids]

# display random images
for i in range(0, len(random_ids)):
    print("Label: %s" % random_labels[i])
    plt.imshow(array_to_img(random_imgs[i,:]).convert('L'), cmap='gray')
    plt.show()

# pre-processing of images
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(x_train)

test = datagen.flow(random_imgs, random_labels)


# import keras model and layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=image_size))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
