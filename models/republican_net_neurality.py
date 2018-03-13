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
from keras.models import load_model
from keras.optimizers import RMSprop
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
        # 'END' char denotes end of sonnet.
        # '\n' denotes end of line.
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
# Generating dicts for converting between ints and chars

# Note that empty char '' is added for training first 40 characters
# in a sonnet.
chars = np.insert(np.unique(all_char), 1, '')
cdict = {c: i for i, c in enumerate(chars)}
cdict_rev = {i: c for i, c in enumerate(chars)}

x_train = []
y_train = []
# Didn't use semi-redundant sequences due to perceived lack of data
# and availability of GPU for faster training.
for sonnet in sonnets:
    for i in range(40):
        blanks = [''] * (40 - i)
        x_train.append([blanks + sonnet[:i]])
        y_train.append(sonnet[i])
    for i in range(len(sonnet) - 40):
        x_train.append([sonnet[i:i + 40]])
        y_train.append(sonnet[i + 40])

# One-hot encoding the labels
for i, y in enumerate(y_train):
    y_train[i] = cdict[y]
y_train = keras.utils.np_utils.to_categorical(y_train)

for x in x_train:
    for x2 in x:
        for i, c in enumerate(x2):
            x2[i] = cdict[c]

np_x = np.array(x_train)

### RNN (Single LSTM layer with ~150 cells)

# First time usage
'''
model = Sequential()
model.add(LSTM(200, input_shape=(1, 40)))

model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.save('../data/currnn.h5')
'''

# Use this for every run after the first
model = load_model('../data/currnn.h5')

# Train the model, iterating on the data in batches of 32 samples
'''
while True:
    model.fit(np_x, y_train, epochs=5, batch_size=32, verbose=1)
    model.save('../data/currnn.h5')
    print('Model saved.')
'''


### Predictions

# helper function for keras' bugged sample function:
# takes elementwise log of a numpy array, and returns -inf
# in case of 0 (instead of just breaking)
def better_log(preds):
    new_p = []
    for p in preds:
        if p == 0:
            new_p.append(-1 * math.inf)
        else:
            new_p.append(np.log(p))
    return np.array(new_p)

# Borrowed from keras repo examples
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = better_log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Change temperatures here
temps = [0.25, 0.6, 0.65, 0.7, 0.75, 1.5]
for t in temps:
    print(t)
    # Seed - previously all letters were capitalized, so this is too.
    new_sonnet = np.array(list('SHALL I COMPARE THEE TO A SUMMER\'S DAY?\n'))
    next_char = ''
    # Sonnet length restricted, since temperature of < 0.5 tends to not end.
    while next_char != 'END' and len(new_sonnet) < 2000:
        sl = len(new_sonnet)
        sonnet_vals = np.zeros((1, 1, 40))
        for i, c in enumerate(new_sonnet[sl - 40:sl]):
            sonnet_vals[0, 0, i] = cdict[c]
        np_test = np.array(sonnet_vals)

        next_val = model.predict(x=np_test)[0]
        next_val2 = sample(next_val, t)

        next_char = cdict_rev[next_val2]
        print(next_char, end='')
        new_sonnet = np.append(new_sonnet, next_char)
    print('\n')

    np.savetxt('../sonnets/rnn_sonnet_' + str(t) + '.txt', new_sonnet, fmt='%s', delimiter='', newline='', comments='')