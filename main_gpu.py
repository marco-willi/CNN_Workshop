# import modules
from keras.preprocessing.image import array_to_img, load_img
import matplotlib.pyplot as plt
import random
import os


# Path to image data
path = 'D:/Studium_GD/Zooniverse/Data/cnn_workshop/PetImages/'
path = '/host/data/cnn_workshop/EleZebra/'

# get class labels
class_labels = os.listdir(path + 'train')

# parameters
image_size_for_training = (250, 250, 3)


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

look_at_imgs(path + class_labels[0] + '/')
look_at_imgs(path + class_labels[1] + '/')


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
    path + 'train/',
    target_size=image_size_for_training[0:2],
    batch_size=1000,
    class_mode='binary',
    seed=123)

X_raw = raw_gen.next()

# fit data generators on raw data
datagen_train.fit(X_raw[0])
datagen_test.fit(X_raw[0])

# fetch data from directory in specified batch sizes
train_generator = datagen_train.flow_from_directory(
    path + 'train/',
    target_size=image_size_for_training[0:2],
    batch_size=batch_size,
    class_mode='binary',
    seed=123)

test_generator = datagen_test.flow_from_directory(
    path + 'test/',
    target_size=image_size_for_training[0:2],
    batch_size=batch_size,
    class_mode='binary',
    seed=123)

# dummy generator to look at data
dummy_gen = datagen_train.flow_from_directory(
    path + 'train/',
    target_size=image_size_for_training[0:2],
    batch_size=5,
    class_mode='binary',
    seed=123)

# lets take a  look what the data generator actually does
data_batch = dummy_gen.next()
for i in range(0, len(data_batch[1])):
    print("Label: %s" % data_batch[1][i])
    plt.imshow(array_to_img(data_batch[0][i, :]))
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
model.add(Conv2D(32, (3, 3), input_shape=image_size_for_training))
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
    epochs=2,
    workers=2,
    validation_data=test_generator,
    validation_steps=test_generator.n // batch_size)

# predict on test sample

# get a test batch
test_batch_data = test_generator.next()
# get class information to make correct mapping
classes = test_generator.class_indices
for cl, no in classes.items():
    if no == 1:
        class1 = cl
# calculate predictions
p_test = model.predict_on_batch(test_batch_data[0])

# show some images and their prediction
for i in range(0, len(test_batch_data[1])):
    print("Predicted %s percent of being a %s" %
          (round(float(p_test[i] * 100), 2), class1))
    plt.imshow(array_to_img(test_batch_data[0][i, :]))
    plt.show()

# Load a pre-trained model
from keras.applications.inception_v3 import InceptionV3

# load Googles inception model with imagenet weights
base_model = InceptionV3(
    include_top=False,
    weights='imagenet',
    input_shape=image_size_for_training)

# take a look at its architecture
base_model.summary()

# set layers to non-trainable
for layer in base_model.layers:
    layer.trainable = False


# add fully connected output layer to model
top_model = base_model.output
top_model = Flatten()(top_model)
top_model = Dense(256, activation='relu')(top_model)
top_model = Dropout(0.5)(top_model)
top_model = Dense(1, activation='sigmoid')(top_model)

# this is the model we will train (pre-trained convoltional part
# and top layer)
model = Model(inputs=base_model.input, outputs=top_model)

# compile model
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# look at model architecture
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
