# coding: utf-8

# In[56]:


from basic_model import base_model as BM
from keras import backend as K
from keras.layers import Input, Activation
from keras.layers.core import Lambda
from keras.models import Model
from keras.callbacks import ModelCheckpoint, TensorBoard
from itertools import combinations
import numpy as np
import os
import cv2

# In[57]:


def l1_distance(inputs):
    input1, input2 = inputs
    print(input1.shape, input2.shape)
    output = K.abs(input1 - input2)
    print(output.shape)
    return output


def l1_distance_output_shape(shapes):
    shape1, shape2 = shapes
    assert shape1 == shape2
    return (1,)


# In[58]:


def siamese_model():
    base_model = BM()
    input1 = Input(shape=(160, 60, 3,))
    input2 = Input(shape=(160, 60, 3,))
    feature_vec1 = base_model(input1)
    feature_vec2 = base_model(input2)
    distance = Lambda(l1_distance, output_shape=l1_distance_output_shape)([feature_vec1, feature_vec2])
    output = Activation('sigmoid')(distance)
    return Model(inputs=[input1, input2], outputs=output)


# In[59]:


def create_input_pairs(X, labels):
    input = zip(X, labels)
    input1, input2 = [], []
    labels = []
    combs = list(combinations(input, 2))
    input1 += [comb[0][0] for comb in combs]
    input2 += [comb[1][0] for comb in combs]
    labels += [comb[0][1] == comb[1][1] for comb in combs]
    input1 = np.array(input1)
    input2 = np.array(input2)
    labels = np.array(labels)
    return input1, input2, labels


# In[60]:


def load_data():
    filenames = os.listdir('./datasets/cuhk01/')
    x = np.array([cv2.imread(os.path.join(os.path.abspath('./datasets/cuhk01/'), filename)) for filename in filenames])
    labels = np.array([int(filename[:4]) for filename in filenames])
    idx = np.argsort(labels)
    x = x[idx]
    labels = np.sort(labels)
    return x[:100], labels[:100]


# In[55]:


if __name__ == '__main__':
    model = siamese_model()
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    X, labels = load_data()
    input1, input2, labels = create_input_pairs(X, labels)
    filepath = './siamese/weights.{epoch:02d}-{val_loss:.2f}.hdf5'
    cpkt1 = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=True, mode='auto', period=1)
    cpkt2 = TensorBoard(log_dir='./siamese/tensorboard', histogram_freq=0, write_graph=True, write_images=True)
    model.summary()
    model.fit([input1, input2], labels, epochs=100, batch_size=20, verbose=2, shuffle=True, validation_split=0.2, callbacks=[cpkt1, cpkt2])


