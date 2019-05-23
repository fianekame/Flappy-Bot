import numpy as np
import keras

from keras.models import model_from_json, Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam
from sklearn.model_selection import train_test_split


_ROOTPATH = "/home/galgadot/Documents/Skripsi/FlappyBot/"
_LEARNING_RATE = 1e-4


""" Build Model """

def createModel():
    print("Now we build the model")
    model = Sequential()
    model.add(Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='same',input_shape=(80,80,4)))  #80*80*4
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 4, 4, subsample=(2, 2), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(2))
    adam = Adam(lr=_LEARNING_RATE)
    model.compile(loss='mse',optimizer=adam)
    print("We finish building the model")
    return model


""" Initializing Data """

DataSize = "5000"
data_x = np.load(_ROOTPATH+'initial_data/'+DataSize+' Data/DataX.npy')
data_y = np.load(_ROOTPATH+'initial_data/'+DataSize+' Data/DataY.npy')

train_x = data_x
train_x.shape
train_y = data_y
train_y.shape

train_x = train_x.astype('float32')
train_x = train_x / 255.
# test_x = test_x.astype('float32')
# test_x = test_x / 255.


""" ================= """

""" Data For Train """
train_x,valid_x,train_label,valid_label = train_test_split(train_x, train_y, test_size=0.2, random_state=13)
train_x.shape,valid_x.shape,train_label.shape,valid_label.shape

batch_size = 32
epochs = 20
num_classes = 2

flappy_model = createModel()
flappy_model.summary()

""" Starting Training Model """
# flappy_train = flappy_model.fit(train_x, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_x, valid_label))

""" Save Model By Name Using JSON """

""" Serialize model to JSON """
model_json = flappy_model.to_json()
with open("saved_networks/saved_"+DataSize+"model.json", "w") as json_file:
    json_file.write(model_json)
""" serialize weights to HDF5 """
flappy_model.save_weights("saved_networks/saved_"+DataSize+"model.h5")
print("Saved model to disk")
