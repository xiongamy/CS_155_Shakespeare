import numpy as np
import random
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

# load data into Keras format
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras import regularizers


### Data retrieval

data = np.genfromtxt('../data/shakespeare.txt', dtype='str', delimiter='\n')

sonnets = []
sonnet = []
for line in data[1:]:
    if len(line) > 3:
        sonnet.append(list(line.upper()))
        sonnet.append(['\n'])
    else:
        sonnet.append(['END'])
        sonar = []
        for l in sonnet:
            sonar += l
        sonnets.append(sonar)
        sonnet = []

# Gets the last sonnet.
sonnet.append(['END'])
sonar = []
for l in sonnet:
    sonar += l
sonnets.append(sonar)

### Data formatting

all_char = []
for s in sonnets:
    all_char += s
chars = np.insert(np.unique(all_char), 1, '')
cdict = {c: i for i, c in enumerate(chars)}

x_train = []
y_train = []
for sonnet in sonnets:
    for i in range(40):
        blanks = [''] * (40 - i)
        x_train.append(blanks + sonnet[:i])
        y_train.append(sonnet[i])
    for i in range(len(sonnet) - 40):
        x_train.append(sonnet[i:i + 40])
        y_train.append(sonnet[i + 40])

# One-hot encoding the labels
for i, y in enumerate(y_train):
    y_train[i] = cdict[y]
y_train = keras.utils.np_utils.to_categorical(y_train)

np_x = np.array(x_train)

### RNN (Single LSTM layer with ~150 cells)

model = Sequential()
model.add(LSTM(150, input_shape=(40,)))

model.add(Dense(len(chars)))
model.add(Activation('softmax'))

# For a multi-class classification problem
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model, iterating on the data in batches of 32 samples
history = model.fit(np_x, y_train, epochs=10, batch_size=32, verbose=0)


### Predictions

new_sonnet = np.full(40, '')
next_char = ''
while next_char != 'END':
    sl = len(new_sonnet)
    next_char = model.predict_classes(x=new_sonnet[s1 - 40:s1])
    new_sonnet.append(next_char)

print(new_sonnet)
'''
np.savetxt("../sonnets/rnn_sonnets.txt", save, fmt='%i', delimiter=',', header='Id,Prediction', comments='')'''