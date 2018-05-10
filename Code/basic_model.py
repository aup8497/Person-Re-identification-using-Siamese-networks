from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPool2D, Activation, BatchNormalization, Flatten, InputLayer
from keras.utils import to_categorical
from keras.optimizers import SGD
import os
import cv2
import numpy as np


def base_model():

    model = Sequential([
        InputLayer(input_shape=(160, 60, 3)),
        Conv2D(64, (3,3), padding='same'),
        BatchNormalization(axis=1, momentum=0.99, epsilon=1e-3),
        Activation('relu'),
        Conv2D(64, (3,3), padding='same'),
        BatchNormalization(axis=1, momentum=0.99, epsilon=1e-3),
        Activation('relu'),
        MaxPool2D(pool_size=(2,2), strides=(2,2)),

        Conv2D(128, (3,3), padding='same'),
        BatchNormalization(axis=1, momentum=0.99, epsilon=1e-3),
        Activation('relu'),
        Conv2D(128, (3,3), padding='same'),
        BatchNormalization(axis=1, momentum=0.99, epsilon=1e-3),
        Activation('relu'),
        MaxPool2D(pool_size=(2,2), strides=(2,2)),

        Conv2D(256, (3,3), padding='same'),
        BatchNormalization(axis=1, momentum=0.99, epsilon=1e-3),
        Activation('relu'),
        Conv2D(256, (3,3), padding='same'),
        BatchNormalization(axis=1, momentum=0.99, epsilon=1e-3),
        Activation('relu'),
        Conv2D(256, (3,3), padding='same'),
        BatchNormalization(axis=1, momentum=0.99, epsilon=1e-3),
        Activation('relu'),
        MaxPool2D(pool_size=(2,2), strides=(2,2)),

        Conv2D(512, (3,3), padding='same'),
        BatchNormalization(axis=1, momentum=0.99, epsilon=1e-3),
        Activation('relu'),
        Conv2D(512, (3,3), padding='same'),
        BatchNormalization(axis=1, momentum=0.99, epsilon=1e-3),
        Activation('relu'),
        Conv2D(512, (3,3), padding='same'),
        BatchNormalization(axis=1, momentum=0.99, epsilon=1e-3),
        Activation('relu'),
        MaxPool2D(pool_size=(2,2), strides=(2,2)),

        Conv2D(512, (3,3), padding='same'),
        BatchNormalization(axis=1, momentum=0.99, epsilon=1e-3),
        Activation('relu'),
        Conv2D(512, (3,3), padding='same'),
        BatchNormalization(axis=1, momentum=0.99, epsilon=1e-3),
        Activation('relu'),
        Conv2D(512, (3,3), padding='same'),
        BatchNormalization(axis=1, momentum=0.99, epsilon=1e-3),
        Activation('relu'),
        MaxPool2D(pool_size=(2,2), strides=(2,2)),

        Flatten(),
        Dense(4096, activation='relu'),
        Dense(4096, activation='relu'),
        Dense(972, activation='relu')
    ])
#    model.summary()

    return model


def load_data():

    filenames = os.listdir('./datasets/cuhk01/')
    x = np.array([cv2.imread(os.path.join(os.path.abspath('./datasets/cuhk01/'), filename)) for filename in filenames])
    labels = np.array([int(filename[:4]) for filename in filenames])
    return x, to_categorical(labels)


def get_feature_vec(model, x, labels):

	model.load_weights('model.h5py')
	new_model = Model(model.layers[0].input, model.layers[-2].output)
	feature_vec = new_model.predict(x)

	return new_model, feature_vec


def cos_sim(model, feature_vec, labels, filename):

    image = np.array(cv2.imread(os.path.join(os.paht.abspath('./datasets/cuhk01/'), filename)))
    input_feature_vec = model.predict(np.expand_dims(image, axis=0))
    similarity = np.sum(feature_vec * input_feature_vec, axis=1)
    similarity = similarity / np.linalg.norm(feature_vec, axis=1)
    similarity = similarity / np.linalg.norm(input_feature_vec)

    return [np.argmax(labels[i]) for i in np.argsort(similarity)[:6]]


if __name__ == '__main__':

    model = base_model()
    x, labels = load_data()
    print(x.shape)
    print(labels.shape)
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    model.fit(x, labels, batch_size=50, epochs=500)
    model.save_weights('model.h5py')
