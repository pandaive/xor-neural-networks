import numpy as np
import random as rand

from keras.models import Sequential
from keras.layers import Dense, Activation

# prepare data
nb_train = 64
data = []
labels = []

for i in range(nb_train):
    x = rand.randint(0, 1)
    y = rand.randint(0, 1)
    noisex = np.random.normal()*(10**-1)
    noisey = np.random.normal()*(10**-1)
    labels.append(1) if x != y else labels.append(0)
    data.append([x+noisex, y+noisey])
    
x_train = np.array(data)
y_train = np.array(labels)

# prepare model
model = Sequential()
# 2 inputs, 2 layers (OR, NAND)
model.add(Dense(2, input_shape=((2,)), activation="tanh"))
# 1 layer - AND
model.add(Dense(1,activation="tanh"))


model.compile(loss='mse', optimizer = 'sgd')

# train - switch verbose to 0 if you don't want tons of output in console
model.fit(x_train,y_train, epochs=5000, batch_size=32, verbose=1, validation_split=0.2, shuffle=True)

# check on actual pairs
X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]])

print(model.predict_proba(X))