import numpy as np

import keras
from keras.models import model_from_json
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from keras.utils import to_categorical

_ROOTPATH = "/home/galgadot/Documents/Skripsi/FlappyBot/"

def loadModel():
    json_file = open(_ROOTPATH+'saved_networks/saved_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(_ROOTPATH+"saved_networks/saved_model.h5")
    adam = Adam(lr=1e-4)
    loaded_model.compile(loss='mse',optimizer=adam)
    print("Loaded model from disk")
    return loaded_model

def getData(dataSize):
    foldername = "Data "+str(dataSize)
    data_x = np.load(_ROOTPATH+'initial_data/'+foldername+'/DataX.npy')
    data_y = np.load(_ROOTPATH+'initial_data/'+foldername+'/DataY.npy')
    print(data_x.shape," - ",data_y.shape)
    return data_x, data_y

myModel = loadModel()
data , target = getData(500)
newTarget = to_categorical(target).tolist()
newTarget[0]
test_eval = myModel.evaluate(data[0], newTarget[0], verbose=1)
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
