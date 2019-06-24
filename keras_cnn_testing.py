import numpy as np
import keras
from keras.models import model_from_json
from keras.optimizers import Adam
from matplotlib import pyplot as plt
%matplotlib inline
import random
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
myModel.summary()
data , target = getData(1000)
target
target = to_categorical(target)
target.shape
target

predicted_classes = myModel.predict(data)
predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
predicted_classes.shape
test_yt = np.argmax(np.round(target),axis=1)
predicted_classes

""" Data Benar """
correct = np.where(predicted_classes==test_yt)[0]
print ("Total %d Benar" % len(correct))
for i, correct in enumerate(correct[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(data[correct], cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], test_yt[correct]))
    plt.tight_layout()

""" Salah Tebak """
incorrect = np.where(predicted_classes!=test_yt)[0]
print ("Total %d Salah" % len(incorrect))
for i, incorrect in enumerate(incorrect[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(data[incorrect], cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], test_yt[incorrect]))
    plt.tight_layout()


from sklearn.metrics import classification_report
target_names = ["Class {}".format(i) for i in range(2)]
target_names
print(classification_report(test_yt, predicted_classes, target_names=target_names))
