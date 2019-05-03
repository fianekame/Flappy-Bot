""" Importing Lib Yang Dibutuhkan """

import cv2
import numpy as np
from matplotlib import pyplot as plt
from keras.utils import to_categorical

""" Import Selesai """

""" Encoding Dan Simpan File Untuk Action One Hot Baru """

action = np.load("../initial_data/ActionFor5000Data.npy")
actonehot = to_categorical(action)
np.save("../initial_data/DataY",np.array(actonehot))

print('Original label:\n', action[5:9])
print('After conversion to one-hot:\n', actonehot[5:9])

""" Akhir Konversi """

""" Get Sample For Staking """

samplefile = "../logs_bird/sample.png";
x_t = cv2.imread(samplefile, cv2.IMREAD_COLOR)
x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

print('Size Before Staking:\n', x_t.shape)
print('Size After  Staking :\n', s_t.shape)

""" End For Sample Staking """

""" Preparing Data """

DataX = []
for i in range(5000):
    foldername = "../logs_bird/";
    filename = foldername+"frame"+str(i)+".png"
    img = cv2.imread(str(filename), cv2.IMREAD_COLOR)
    x_t1 = cv2.cvtColor(cv2.resize(img, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
    x_t1 = np.reshape(x_t1, (80, 80, 1))
    s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

    DataX.append(s_t1)
np.save("../initial_data/DataX",np.array(DataX))

""" End For Preparing Data """
