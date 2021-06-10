import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.python.keras.applications.resnet import ResNet50
from tensorflow.keras import backend as K

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

df = pd.read_csv("fer2013.csv")
train, validate, test = np.split(df.sample(frac=1), [int(.6 * len(df)), int(.8 * len(df))])

# evaluate = df.sample(frac=0.20, random_state=200)  # random state is a seed value
# df = df.drop(df.index)
# train = df.sample(frac=0.8, random_state=200)  # random state is a seed value
# test = df.drop(train.index)
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
        i = Image.frombuffer('L', (48, 48), b)
        i.save("train/" + str(emotion) + "/" + str(count) + '.png')
        count += 1
    count = 1
    for idx, row in test.iterrows():
        emotion = row['emotion']
        pixel = row['pixels']
        usage = row['Usage']
        b = bytes(int(p) for p in pixel.split())
        i = Image.frombuffer('L', (48, 48), b)
        i.save("test/" + str(emotion) + "/" + str(count) + '.png')
        count += 1

# from keras.applications.vgg16 import VGG16

model = tf.keras.applications.vgg16.VGG16(weights='imagenet')
print(model.summary())

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

ResNet50_model = ResNet50(weights='imagenet', include_top=False, input_shape=(48, 48, 3), classes=7)

for layers in ResNet50_model.layers:
    layers.trainable = True
def swish_activation(x):
    return (K.sigmoid(x) * x)
opt = tf.keras.optimizers.SGD(lr=0.01, momentum=0.7)
resnet50_x = Flatten()(ResNet50_model.output)
resnet50_x = Dense(256, activation=swish_activation, bias_regularizer=tf.keras.regularizers.l2(0.01))(resnet50_x)
resnet50_x = Dense(7, activation='softmax')(resnet50_x)
resnet50_x_final_model = tf.keras.Model(inputs=ResNet50_model.input, outputs=resnet50_x)
resnet50_x_final_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])

number_of_epochs = 10
resnet_filepath = 'resnet50_model.hdf5'
resnet_checkpoint = tf.keras.callbacks.ModelCheckpoint(resnet_filepath, monitor='val_acc', verbose=1,
                                                       save_best_only=True, mode='max')
resnet_early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.05, patience=5, min_lr=0.000002)
callbacklist = [resnet_checkpoint, resnet_early_stopping, reduce_lr]
resnet50_history = resnet50_x_final_model.fit(train_generator, epochs=number_of_epochs,
                                              validation_data=validation_generator,
                                              # steps_per_epoch=train_generator.samples // train_generator.batch_size,
                                              # validation_steps=validation_generator.samples // validation_generator.batch_size,
                                              callbacks=callbacklist, verbose=1)


N = np.arange(0, number_of_epochs)
plt.figure()
plt.plot(N, resnet50_history.history["loss"], label="train_loss")
plt.plot(N, resnet50_history.history["val_loss"], label="val_loss")
plt.plot(N, resnet50_history.history["acc"], label="train_acc")
plt.plot(N, resnet50_history.history["val_acc"], label="val_acc")
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
