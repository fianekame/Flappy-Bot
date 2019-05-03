import numpy as np

import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

from sklearn.model_selection import train_test_split

_ROOTPATH = "/home/galgadot/Documents/Skripsi/FlappyBot/"

""" Initializing Data """

data_x = np.load('initial_data/DataX.npy')
data_y = np.load('initial_data/DataY.npy')

train_x , test_x = data_x[:500], data_x[500:600]
train_x.shape, test_x.shape
train_y , test_y = data_y[:500], data_y[500:600]
train_y.shape, test_y.shape

train_x = train_x.astype('float32')
test_x = test_x.astype('float32')
train_x = train_x / 255.
test_x = test_x / 255.

""" ================= """

""" Data For Train """
train_x,valid_x,train_label,valid_label = train_test_split(train_x, train_y, test_size=0.2, random_state=13)
train_x.shape,valid_x.shape,train_label.shape,valid_label.shape

batch_size = 64
epochs = 100
num_classes = 2

flappy_model = Sequential()
flappy_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(80,80,4)))
flappy_model.add(LeakyReLU(alpha=0.1))
flappy_model.add(MaxPooling2D((2, 2),padding='same'))
flappy_model.add(Dropout(0.25))

flappy_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
flappy_model.add(LeakyReLU(alpha=0.1))
flappy_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
flappy_model.add(Dropout(0.25))

flappy_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
flappy_model.add(LeakyReLU(alpha=0.1))
flappy_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
flappy_model.add(Dropout(0.4))

flappy_model.add(Flatten())
flappy_model.add(Dense(128, activation='linear'))
flappy_model.add(LeakyReLU(alpha=0.1))
flappy_model.add(Dropout(0.3))

flappy_model.add(Dense(num_classes, activation='softmax'))
flappy_model.summary()

flappy_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
flappy_train = flappy_model.fit(train_x, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_x, valid_label))

# serialize model to JSON
model_json = flappy_model.to_json()
with open("saved_networks/saved_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
flappy_model.save_weights("saved_networks/saved_model.h5")
print("Saved model to disk")
