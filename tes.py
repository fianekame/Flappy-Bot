import numpy as np
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import model_from_json

_ROOTPATH = "/home/galgadot/Documents/Skripsi/FlappyBot/"

""" Initializing Data """

data_x = np.load(_ROOTPATH+'initial_data/DataX.npy')
data_y = np.load(_ROOTPATH+'initial_data/DataY.npy')

train_x , test_x = data_x[:500], data_x[500:600]
train_x.shape, test_x.shape
train_y , test_y = data_y[:500], data_y[500:600]
train_y.shape, test_y.shape

train_x = train_x.astype('float32')
test_x = test_x.astype('float32')
train_x = train_x / 255.
test_x = test_x / 255.

LEARNING_RATE = 1e-4
checkpoint = tf.train.get_checkpoint_state("saved_networks")
# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint,save_weights_only=True,verbose=1)

def create_model():
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
    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse',optimizer=adam)
    print("We finish building the model")
    return model

checkpoint_path = _ROOTPATH+"saved_networks/mymodel.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# model = create_model()
# model.fit(train_x, train_y, epochs = 10, validation_data = (test_x,test_y), callbacks = [cp_callback])

model1 = create_model()
model1.load_weights(checkpoint_path)
predicted_classes = model1.predict(test_x)
predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
predicted_classes
# loss, acc = model1.evaluate(test_x, test_y)
# print("Restored model, accuracy: {:5.2f}%".format(100*acc))
