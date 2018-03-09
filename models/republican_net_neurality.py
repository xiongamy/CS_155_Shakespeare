import numpy as np
import random
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

# load data into Keras format
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras import regularizers

train_data = np.loadtxt('../data/shakespeare.txt', delimiter=' ', skiprows=1)

gen = 0

#while True:
#grand = np.loadtxt('../data/nnh.txt', delimiter=' ', skiprows=1)
gen += 1
#h = random.randint(1, 1000)
#n1 = random.randint(1, 7)
#d = random.uniform(0.0, 0.3)
#n2 = random.randint(1, 50)
#e = random.randint(4, 15)

# Evolutionary training of hyperparams

#r_point = farthest_in_convex(grand[:50, :-1])
#r_point = grand[0]
#r_point = [1000, 3, 0.15, 20, 5]

h = 975
n1 = 8
d = 0.13
n2 = 21
e = 4

'''
h = int(grand[c, 0])
hr = random.randint(max(1 - h, -10), min(1000 - h, 10))
h += hr

n1 = int(grand[c, 1])
n1r = random.randint(max(1 - n1, -2), 2)
n1 += n1r

d = grand[c, 2]
dr = random.uniform(max(-d, -0.03), min(1 - d, 0.03))
d += dr

n2 = int(grand[c, 3])
n2r = random.randint(max(1 - n2, -5), 5)
n2 += n2r

e = int(grand[c, 4])
er = random.randint(max(1 - e, -2), 2)
e += er
'''

y_train = train_data[:, 0]
x_temp = train_data[:, 1:]

x_train = np.zeros((len(x_temp), h))
count = 0
while count < h:
    for i in range(len(x_temp)):
        x_train[i, count] = x_temp[i, count]
    count += 1

x_temp = test_data[:, 1:]

x_test = np.zeros((len(x_temp), h))
count = 0
while count < h:
    for i in range(len(x_temp)):
        x_test[i, count] = x_temp[i, count]
    count += 1
print(x_test.shape)
# we'll need to one-hot encode the labels
y_train = keras.utils.np_utils.to_categorical(y_train)


acc = 0
kf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(x_train):
    x_tr, x_te = x_train[train_index], x_train[test_index]
    y_tr, y_te = y_train[train_index], y_train[test_index]
    model = Sequential()
    model.add(Dense(n1, input_shape=(h,)))
    model.add(Activation('relu'))
    model.add(Dropout(d))
    model.add(Dense(n2))
    model.add(Activation('relu'))

    model.add(Dense(2))
    model.add(Activation('softmax'))

    # For a multi-class classification problem
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model, iterating on the data in batches of 32 samples
    history = model.fit(x_tr, y_tr, epochs=e, batch_size=32, verbose=0)

    x_predict = model.predict_classes(x=x_te)
    e_out = 0.0
    for i in range(len(x_predict)):
        if (x_predict[i] != y_te[i, 1]):
            e_out += 1
    acc += e_out / len(x_predict)
acc /= 5
print(acc)

'''
print("Iteration: ", gen)
print(grand[0:3])
print("...")
print(grand[49], "\n")
grand = np.concatenate((grand, [[h, n1, d, n2, e, acc]]))
grand = sorted(grand, key=lambda x: x[5])

np.savetxt("../data/nnh.txt", grand, fmt='%f', delimiter=' ', header='h n1 d n2 e err', comments='')
'''
#print(acc)

y_test = model.predict_classes(x=x_test)

ids = np.linspace(1, len(y_test), num=len(y_test))

save = np.concatenate((np.transpose([ids]), np.transpose([y_test])), axis=1)

np.savetxt("../submissions/nn_submission1.csv", save, fmt='%i', delimiter=',', header='Id,Prediction', comments='')
