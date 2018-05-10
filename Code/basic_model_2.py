from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPool2D, Activation, BatchNormalization, Flatten, InputLayer
from keras.utils import to_categorical
from keras.optimizers import SGD
import os
import cv2
import numpy as np
from keras.callbacks import ModelCheckpoint,TensorBoard
from numpy import dot
from numpy.linalg import norm

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
    # model.summary()

    return model


def load_data():

    filenames = os.listdir('./datasets/cuhk01/')
    filenames.sort()
    x = np.array([cv2.imread(os.path.join(os.path.abspath('./datasets/cuhk01/'), filename)) for filename in filenames])
    labels = np.array([int(filename[:4]) for filename in filenames])
    print(labels[:10])
    return x, to_categorical(labels),labels


def get_feature_vec(model, x, labels):

    model.load_weights('model_akshay_before.hdf5')
    new_model = Model(model.layers[0].input, model.layers[-2].output)
    feature_vec = new_model.predict(x)
    new_model.summary()
    return new_model, feature_vec


def cos_sim(model, feature_vec, labels, filename):
    print("the image file name given is "+filename)
    image = np.array(cv2.imread(os.path.join(os.path.abspath('./datasets/cuhk01/'), filename)))
    input_feature_vec = np.array(model.predict(np.expand_dims(image, axis=0)),dtype='float64')

    sim_list = []

    temp_feature_vec = []
    for i in range(len(feature_vec)):
        if (i+1)%4  != 0 :
            temp_feature_vec.append(feature_vec[i])

    print("the number of the feature vectors are ",len(feature_vec))
    print("the number of the feature vectors after removing every 4th image is ",len(temp_feature_vec))

    for feature in np.array(temp_feature_vec,dtype='float64'):
        a = feature
        b = input_feature_vec
        cos_sim = dot(b, a)/(norm(a)*norm(b))
        sim_list.append(cos_sim)

    # similarity = np.sum(feature_vec * input_feature_vec, axis=1)
    # similarity = similarity / np.linalg.norm(feature_vec, axis=1)
    # similarity = similarity / np.linalg.norm(input_feature_vec)
    # print(sim_list)
    # labels_arr = [np.argmax(i) for i in labels]
    # print()
    sim_list_labelled = sorted([(i,j) for i,j in zip(sim_list,labels)],reverse=True)
    sim_list_str = "".join(str(i)+"-"+str(j)+"\n" for i,j in sim_list_labelled)
    f = open('sim_list.txt','w+')
    f.write(sim_list_str)
    f.close()
    # return [np.argmax(labels[i]) for i in np.flip(np.argsort(np.array(sim_list)), axis=0)[:6]]
    # return labels[sim_list.index(max(sim_list))].argmax()
    return sim_list_labelled[:6]

if __name__ == '__main__':

    # index_of_the_test_image = input()
    model = base_model()
    x, categorical_labels, labels = load_data()
    new_model,feature_vec = get_feature_vec(model, x, labels)
    print("The top similarity matched images are (similarity,label)",cos_sim(new_model, feature_vec, labels, '0001004.png'))
