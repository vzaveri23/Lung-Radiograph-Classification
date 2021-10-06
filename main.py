import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import imread

path = "../Disease-classification"
covid = path+"/COVID/"
normal = path+"/Normal/"
opacity = path+"/Lung_Opacity/"
pneumonia = path+"/Viral Pneumonia/"

image_shape=(299, 299, 1)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
image_gen = ImageDataGenerator(width_shift_range=0.1,
                              height_shift_range=0.1,
                              shear_range=0.1,
                              zoom_range=0.1,
                              fill_mode="nearest",
                              validation_split=0.25)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, Reshape
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), input_shape=image_shape, activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(32, kernel_size=(3, 3), input_shape=image_shape, activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), input_shape=image_shape, activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), input_shape=image_shape, activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), input_shape=image_shape, activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(4, activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=2, verbose=1)

batch_size=32
train_gen = image_gen.flow_from_directory(path, target_size=image_shape[:2], color_mode="grayscale",
                                         batch_size=batch_size, class_mode="categorical",
                                          classes=["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"])

test_gen = image_gen.flow_from_directory(batch_size=batch_size,
                                                 directory=path,
                                                 shuffle=False,
                                                 color_mode="grayscale",
                                                 target_size=image_shape[:2],
                                                 subset="validation",
                                                 class_mode='categorical',
                                         classes=["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"])

model.fit(train_gen, validation_data=test_gen, epochs=10, callbacks=[early_stop])