import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.datasets import fashion_mnist
from keras.utils.np_utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPool2D,Dense,Flatten,Dropout,Input,Conv2D,MaxPooling2D,BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from PIL import Image
from pathlib import Path

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

y_gan_list= []

directory = "../GAN/generated/"

filelist = []

for num_class in range(10):
    for num_image in range(100):
        file_name = directory+str(num_class)+"_"+str(num_image)+".png"

        fashion_image = Image.open(file_name)

        resized_image = fashion_image.resize((28, 28))

        filelist.append(np.array(resized_image))

        y_gan_list.append(num_class)


X_gan = np.array(filelist)

print(X_gan)

y_gan = np.array(y_gan_list)

print(X_gan.shape)
print(y_gan.shape)
print(X_gan[1, :, :, 0].shape)
print(X_gan[1, :, :, 0])


## Load the fashion dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

## Reshape the training data 
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))
X_gan = X_gan[:, :, :, 0].reshape((X_gan.shape[0], 28, 28, 1))

## Normalize x train and x test images
X_train = X_train.astype('float') / 255
X_test = X_test.astype('float') / 255
X_gan = X_gan.astype('float') / 255

## Create one hot encoding vectors for y train and y test
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
y_gan = to_categorical(y_gan, num_classes=10)

## Define the model
model = Sequential()

## Add a convolutional layer with 32 filters, 3x3 kernel, relu activation, he uniform kernel initializer, same padding and input shape
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu',
                    kernel_initializer='he_uniform', padding='same', input_shape=(28, 28, 1)))


## Add a batch normalization layer
model.add(BatchNormalization())


## Add a convolutional layer with 32 filters, 3x3 kernel, relu activation, he uniform kernel initializer, and same padding
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
                    kernel_initializer='he_uniform', padding='same'))


## Add a batch normalization layer
model.add(BatchNormalization())


## Add a max pooling 2d layer with 2x2 size
model.add(MaxPooling2D(pool_size=(2, 2)))


## Add dropout layer of 0.2
model.add(Dropout(0.2))


## Add a convolutional layer with 64 filters, 3x3 kernel, relu activation, he uniform kernel initializer, and same padding
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu',
                    kernel_initializer='he_uniform', padding='same'))


## Add a batch normalization layer
model.add(BatchNormalization())


## Add a convolutional layer with 64 filters, 3x3 kernel, relu activation, he uniform kernel initializer, and same padding
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu',
                    kernel_initializer='he_uniform', padding='same'))


## Add a batch normalization layer
model.add(BatchNormalization())


## Add a max pooling 2d layer with 2x2 size
model.add(MaxPooling2D(pool_size=(2, 2)))


## Add dropout layer of 0.2
model.add(Dropout(0.2))


## Add a convolutional layer with 128 filters, 3x3 kernel, relu activation, he uniform kernel initializer, and same padding
model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu',
                    kernel_initializer='he_uniform', padding='same'))


## Add a batch normalization layer
model.add(BatchNormalization())


## Add a convolutional layer with 128 filters, 3x3 kernel, relu activation, he uniform kernel initializer, and same padding
model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu',
                    kernel_initializer='he_uniform', padding='same'))


## Add a batch normalization layer
model.add(BatchNormalization())


## Add a max pooling 2d layer with 2x2 size
model.add(MaxPooling2D(pool_size=(2, 2)))


## Add dropout layer of 0.2
model.add(Dropout(0.2))


## Flatten the resulting data
model.add(Flatten())


## Add a dense layer with 128 nodes, relu activation and he uniform kernel initializer
model.add(Dense(units=128, activation='relu', kernel_initializer='he_uniform'))


## Add a batch normalization layer
model.add(BatchNormalization())


## Add dropout layer of 0.2
model.add(Dropout(0.2))


## Add a dense softmax layer
model.add(Dense(units=10, activation='softmax'))


## Set up early stop training with a patience of 3
early_stop = EarlyStopping(patience=3)


## Compile the model with adam optimizer, categorical cross entropy and accuracy metrics
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

## Define the number of steps to take per epoch as training examples over 64
steps_per_epoch = len(X_train) / 64
epochs=200

print(X_train.shape)

## Fit the model with the generated data, 200 epochs, steps per epoch and validation data defined.
fitted_model = model.fit(X_train, y_train, epochs=epochs, steps_per_epoch=steps_per_epoch, validation_data=(X_test, y_test), callbacks=[early_stop])

#Show graph of loss over time for training data
num = range(1, len(fitted_model.history['loss'])+1)
plt.plot(num, fitted_model.history['loss'], '-b', label='loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss over time for training data")
plt.show()

#Show graph of accuracy over time for training data
num = range(1, len(fitted_model.history['loss'])+1)
plt.plot(num, fitted_model.history['accuracy'], '-b', label='accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy over time for training data")
plt.show()

print("Accuracy for Fashion_MNIST testset")
print("Accuracy : %.2f %%" % (model.evaluate(X_test, y_test)[1]))

print(" ")
print("Accuracy for GAN Generated Images")
print("Accuracy : %.2f %%" % (model.evaluate(X_gan, y_gan)[1]))
