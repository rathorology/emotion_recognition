import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten,BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import  Adam
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def swish_activation(x):
    return (K.sigmoid(x) * x)


df = pd.read_csv("fer2013.csv")
X_train, train_y, X_test, test_y = [], [], [], []

for index, row in df.iterrows():
    val = row['pixels'].split(" ")
    try:
        if 'Training' in row['Usage']:
            X_train.append(np.array(val, 'float32'))
            train_y.append(row['emotion'])
        elif 'PublicTest' in row['Usage']:
            X_test.append(np.array(val, 'float32'))
            test_y.append(row['emotion'])
    except:
        print(f"error occured at index :{index} and row:{row}")

num_labels = 7
batch_size = 128
epochs = 20
width, height = 48, 48

X_train = np.array(X_train, 'float32')
train_y = np.array(train_y, 'float32')
X_test = np.array(X_test, 'float32')
test_y = np.array(test_y, 'float32')

train_y = tf.keras.utils.to_categorical(train_y, num_classes=num_labels)
test_y = tf.keras.utils.to_categorical(test_y, num_classes=num_labels)

# cannot produce
# normalizing data between oand 1
X_train -= np.mean(X_train, axis=0)
X_train /= np.std(X_train, axis=0)

X_test -= np.mean(X_test, axis=0)
X_test /= np.std(X_test, axis=0)

X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)

X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)

model = Sequential()

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(X_train.shape[1:])))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

# 2nd convolution layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

# 3rd convolution layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())

model.add(Dense(64, activation=swish_activation, bias_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(Dropout(0.4))

model.add(Dense(num_labels, activation='softmax'))

model.summary()
# initialize the model and optimizers
optimizer = Adam(learning_rate=0.01)
# Compliling the model
model.compile(loss="categorical_crossentropy",
              optimizer=optimizer,
              metrics=['accuracy'])
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
vgg16_filepath = 'model.hdf5'
checkpoint = tf.keras.callbacks.ModelCheckpoint(vgg16_filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                                mode='min')
# Training the model
history = model.fit(X_train, train_y,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    callbacks=[early_stopping, checkpoint],
                    validation_data=(X_test, test_y),
                    shuffle=True)
N = np.arange(0, epochs)
plt.figure()
plt.plot(N, history.history["loss"], label="train_loss")
plt.plot(N, history.history["val_loss"], label="val_loss")
plt.plot(N, history.history["accuracy"], label="train_acc")
plt.plot(N, history.history["val_accuracy"], label="val_acc")
plt.title("Training/Validation Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig('sequential_plot.png')
plt.show()
