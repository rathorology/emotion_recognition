# import the necessary packages
import logging
import os
import random

import cv2
import numpy as np
from skimage.io import imread_collection
# disable the warnings
from sklearn.metrics import classification_report
from tensorflow.keras import backend as K

logging.disable(logging.WARNING)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#
# # function to process the image similar to the training process
# def process_image(image_path):
#     # load the input image and resize it to the target spatial dimensions
#     image = cv2.imread(image_path)
#     width, height = 48, 48
#
#     image = cv2.resize(image, (width, height))
#
#     # image to array
#     b = tf.keras.preprocessing.image.img_to_array(image)
#
#     # normalize the image
#     processed_image = np.array(b, dtype="float") / 255.0
#
#     return processed_image
#
#
# def make_prediction(image_path, model_path):
#     collection_directory = str(image_path) + '/*.png'
#
#     print(collection_directory)
#     # creating a collection with the available images
#     all_frames = imread_collection(collection_directory)
#     # process the image
#     image = process_image(image_path)
#
#     # read the image for the output
#     output = cv2.imread(image_path)
#
#     # load the model and label binarizer from the directory
#     print("[INFO] loading model and label binarizer...")
#
#     # # relative paths to the model and labels
#     # model_path = os.path.join(model_dir, 'trained_VGG_model.h5')
#     # label_file_path = os.path.join(model_dir, 'labels')
#
#     # load the model and the label encoder
#     model = load_model(model_path)
#     # lb = pickle.loads(open(label_file_path, "rb").read())
#
#     # make a prediction on the image
#     image = np.expand_dims(image, axis=0)
#     pred_result = model.predict(image)
#
#     # extract the class label which has the highest corresponding probability
#     i = pred_result.argmax(axis=1)[0]
#     label = {"0": "angry", "1": "disgust", "2": "fear", "3": "happy", "4": "sad", "5": "surprise", "6": "neutral"}
#     print(i)
#     # draw the class label + probability on the output image
#     text = "{}: {:.2f}%".format(label, pred_result[0][i] * 100)
#     cv2.putText(output, text, (5, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 4)
#
#     # display the result on the screen
#     print("Predicted label = {}: {:.2f}%".format(label[str(i)], pred_result[0][i] * 100))
#
#     # # save the output image with label
#     # cv2.imwrite('output.jpg', output)
#
#     # evaluate the network
#     predictions = model.predict(testX, batch_size=32)
#     print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=label))
#

def swish_activation(x):
    return (K.sigmoid(x) * x)

if __name__ == '__main__':
    import tensorflow as tf

    from imutils import paths

    data = []
    labels = []
    label_names = {"0": "angry", "1": "disgust", "2": "fear", "3": "happy", "4": "sad", "5": "surprise", "6": "neutral"}

    # grab the image paths and shuffle them
    imagePaths = sorted(list(paths.list_images("/home/machine/PycharmProjects/emotion_recongnition/validate")))
    random.seed(2)
    random.shuffle(imagePaths)

    IMAGE_WIDTH, IMAGE_HEIGHT = 48, 48

    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
        # image to array
        image = tf.keras.preprocessing.image.img_to_array(image)

        # normalize the image
        image = np.array(image, dtype="float")

        # append the image to the data list
        data.append(image)
        # extract label from the image path and update the labels list
        label = imagePath.split(os.path.sep)[-2]
        labels.append(int(label))
    model = tf.keras.models.load_model("/home/machine/PycharmProjects/emotion_recongnition/model.hdf5",custom_objects={'swish_activation': swish_activation})
    # evaluate the network
    predictions = model.predict(np.array(data), batch_size=32)
    print(classification_report(labels, predictions.argmax(axis=1), target_names=label_names))
