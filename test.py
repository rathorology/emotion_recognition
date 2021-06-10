# import the necessary packages
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
# disable the warnings
from sklearn.metrics import classification_report
from tensorflow.keras import backend as K

logging.disable(logging.WARNING)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

num_labels = 7


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

if __name__ == '__main__':
    import tensorflow as tf
    import seaborn as sns

    label_names = {"0": "angry", "1": "disgust", "2": "fear", "3": "happy", "4": "sad", "5": "surprise", "6": "neutral"}

    model = tf.keras.models.load_model("/home/machine/PycharmProjects/emotion_recongnition/model.hdf5",
                                       custom_objects={'swish_activation': swish_activation})
    # evaluate the network
    predictions = model.predict(np.array(X_test), batch_size=32)
    print(predictions)
    print("==================================================================="
          )
    print(test_y)
    cr = classification_report(test_y.argmax(axis=1), predictions.argmax(axis=1), target_names=label_names,
                               output_dict=True)
    print(cr)
    sns.heatmap(pd.DataFrame(cr).iloc[:-1, :].T, annot=True)
    plt.savefig('Classification_repot_plot.png')
    plt.show()
    # plot_classification_report(cr)
