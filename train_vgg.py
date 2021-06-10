import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

df = pd.read_csv("fer2013.csv")
train, validate, test = np.split(df.sample(frac=1), [int(.6 * len(df)), int(.8 * len(df))])

if not os.path.exists("train"):
    os.makedirs("train")
    os.makedirs("train/0")
    os.makedirs("train/1")
    os.makedirs("train/2")
    os.makedirs("train/3")
    os.makedirs("train/4")
    os.makedirs("train/5")
    os.makedirs("train/6")

    os.makedirs("test")
    os.makedirs("test/0")
    os.makedirs("test/1")
    os.makedirs("test/2")
    os.makedirs("test/3")
    os.makedirs("test/4")
    os.makedirs("test/5")
    os.makedirs("test/6")

    os.makedirs("validate")
    os.makedirs("validate/0")
    os.makedirs("validate/1")
    os.makedirs("validate/2")
    os.makedirs("validate/3")
    os.makedirs("validate/4")
    os.makedirs("validate/5")
    os.makedirs("validate/6")

    count = 1
    for idx, row in train.iterrows():
        emotion = row['emotion']
        pixel = row['pixels']
        usage = row['Usage']

        b = bytes(int(p) for p in pixel.split())
        i = Image.frombuffer('L', (48, 48), b, 'raw', "L", 0, 1)
        # i = tf.keras.preprocessing.image.img_to_array(i)
        #
        # # normalize the image
        # i = np.array(i, dtype="float") / 255.0
        i.save("train/" + str(emotion) + "/" + str(count) + '.png')
        count += 1
    count = 1
    for idx, row in test.iterrows():
        emotion = row['emotion']
        pixel = row['pixels']
        usage = row['Usage']
        b = bytes(int(p) for p in pixel.split())
        i = Image.frombuffer('L', (48, 48), b, 'raw', "L", 0, 1)

        # i = tf.keras.preprocessing.image.img_to_array(i)
        #
        # # normalize the image
        # i = np.array(i, dtype="float") / 255.0
        i.save("test/" + str(emotion) + "/" + str(count) + '.png')
        count += 1
    count = 1
    for idx, row in validate.iterrows():
        emotion = row['emotion']
        pixel = row['pixels']
        usage = row['Usage']
        b = bytes(int(p) for p in pixel.split())
        i = Image.frombuffer('L', (48, 48), b, 'raw', "L", 0, 1)

        #
        # # normalize the image
        # i = np.array(i, dtype="float") / 255.0
        i.save("validate/" + str(emotion) + "/" + str(count) + '.png')
        count += 1

# from keras.applications.vgg16 import VGG16

model = tf.keras.applications.vgg16.VGG16(weights='imagenet')
print(model.summary())


def swish_activation(x):
    return (K.sigmoid(x) * x)


train_datagen = ImageDataGenerator(rescale=1.0 / 255)

# we are rescaling by 1.0/255 to normalize the rgb values if they are in range 0-255 the values are too high for good model performance.
train_generator = train_datagen.flow_from_directory("train",
                                                    batch_size=128,
                                                    shuffle=True,
                                                    class_mode='categorical',
                                                    target_size=(48, 48))

validation_datagen = ImageDataGenerator(
    rescale=1.0 / 255)  # we are only normalising to make the prediction, the other parameters were used for agumentation and train weights
validation_generator = validation_datagen.flow_from_directory("test", shuffle=True, batch_size=128,
                                                              class_mode='categorical', target_size=(48, 48))

vgg16_model = tf.keras.applications.vgg16.VGG16(pooling='avg', weights='imagenet', include_top=False,
                                                input_shape=(48, 48, 3))
for layers in vgg16_model.layers:
    layers.trainable = False
last_output = vgg16_model.layers[-1].output
vgg_x = tf.keras.layers.Flatten()(last_output)
vgg_x = tf.keras.layers.Dense(64, activation=swish_activation,
                              # bias_regularizer=tf.keras.regularizers.l2(0.01)
                              )(vgg_x)
# vgg_x = tf.keras.layers.Dropout(0.4)(vgg_x)
vgg_x = tf.keras.layers.Dense(7, activation='softmax')(vgg_x)
vgg16_final_model = tf.keras.Model(vgg16_model.input, vgg_x)

# initialize the model and optimizers
opt = tf.keras.optimizers.Adam(0.001)
vgg16_final_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])

# VGG16
number_of_epochs = 50
vgg16_filepath = 'vgg_16_model.hdf5'
vgg_checkpoint = tf.keras.callbacks.ModelCheckpoint(vgg16_filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                                    mode='max')
vgg_early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
vgg16_history = vgg16_final_model.fit(train_generator, epochs=number_of_epochs, validation_data=validation_generator,
                                      steps_per_epoch=train_generator.samples // train_generator.batch_size,
                                      validation_steps=validation_generator.samples // validation_generator.batch_size,
                                      callbacks=[vgg_checkpoint, vgg_early_stopping], verbose=1)

N = np.arange(0, number_of_epochs)
plt.figure()
plt.plot(N, vgg16_history.history["loss"], label="train_loss")
plt.plot(N, vgg16_history.history["val_loss"], label="val_loss")
plt.plot(N, vgg16_history.history["acc"], label="train_acc")
plt.plot(N, vgg16_history.history["val_acc"], label="val_acc")
plt.title("Training/Validation Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
plt.savefig('vggnet_plot.png')
#
# acc = vgg16_history.history['accuracy']
# val_acc = vgg16_history.history['val_accuracy']
#
# loss = vgg16_history.history['loss']
# val_loss = vgg16_history.history['val_loss']
