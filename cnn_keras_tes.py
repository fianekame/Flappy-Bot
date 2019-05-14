import numpy as np

import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

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
test_y
# np.reshape(test_x[0], (80, 80, 1))

""" ================= """
batch_size = 64
epochs = 100
num_classes = 2

# load json and create model
json_file = open('saved_networks/saved_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("saved_networks/saved_model.h5")
print("Loaded model from disk")

loaded_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
test_eval = loaded_model.evaluate(test_x, test_y, verbose=1)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

def onehot_to_int(v):
    print(v)
    # return v.tolist().index(1.0)

predicted_classes = loaded_model.predict(test_x)
predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
predicted_classes
test_yt = np.argmax(np.round(test_y),axis=1)
test_yt
correct = np.where(predicted_classes==test_yt)[0]
correct

print(predicted_classes)
print(test_yt)
# for i, correct in enumerate(correct[:9]):
#     print(i)
#     plt.subplot(3,3,i+1)
#     plt.imshow(test_x[correct], cmap='gray', interpolation='none')
#     plt.title("Predicted {}, Class {}".format(predicted_classes[correct], test_yt[correct]))
#     plt.tight_layout()
# plt.show()
# predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
# correct = np.where(predicted_classes==test_y)[0]
# afteronehot = []
# for i in range (len(test_y)):
#     print("Real Data : ", onehot_to_int(test_y[i]))
#     print("Predicted Data : ", onehot_to_int(predicted_classes[i]))
