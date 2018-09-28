# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 12:01:00 2018

@author: todd
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils.data_utils import get_file
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Input, Embedding, Reshape, merge, LSTM, Bidirectional
from keras.layers import SimpleRNN, TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv1D, MaxPooling1D, ZeroPadding1D
from keras.utils import np_utils
from keras.optimizers import Adam
import pickle
import bcolz
import re
from numpy.random import random, permutation, randn, normal, uniform, choice

path = get_file ('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")

text = open(path).read()

chars = sorted(list(set(text)))

chars.insert(0, '\0')

char_to_index = {v:i for i,v in enumerate(chars)}
index_to_char = {i:v for i,v in enumerate(chars)}

total_index = [char_to_index[char] for char in text]

pred_num = 25

xin = [[total_index[j+i] for j in range(0, len(total_index)-1-pred_num, pred_num)] for i in range(pred_num)]

y = [total_index[i+pred_num] for i in range(0, len(total_index)-1-pred_num, pred_num)]

#xin = [[j+i for j in range(0, len(total_index)-1-pred_num, pred_num)] for i in range(pred_num)]
#
#y = [i+pred_num for i in range(0, len(total_index)-1-pred_num, pred_num)]
X = [np.stack(xin[i][:-2]) for i in range(pred_num)]

Y = np.stack(y[:-2])

hidden_layers = 256
vocab_size = 85
n_fac = 42

model = Sequential([
        Embedding(vocab_size, n_fac, input_length = pred_num), #(85, 42, 25)
        SimpleRNN(hidden_layers, activation='relu'),  #（256， 
        Dense(vocab_size, activation='softmax')  #（85,
    ])

model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam())

model.fit(np.stack(X, 1), Y, batch_size=64, epochs=5)

model.save_weights('simpleRNN_3pred.h5')

model.load_weights('simpleRNN_3pred.h5')

model.save_weights('simpleRNN_7pred.h5')

model.load_weights('simpleRNN_7pred.h5')

def predict_next_char(inp):
    index = [char_to_index[i] for i in inp]
    arr = np.expand_dims(np.array(index), axis=0)
    prediction = model.predict(arr)
    return index_to_char[np.argmax(prediction)]

label = predict_next_char('those w')