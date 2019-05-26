""" Importing Lib Yang Dibutuhkan """

import numpy as np
import skimage as skimage
import collections

from matplotlib import pyplot as plt
from skimage import transform, color, exposure
from skimage.io import imsave, imread, imshow
from keras.utils import to_categorical

""" Import Selesai """

""" Encoding Dan Simpan File Untuk Action One Hot Baru """

# action = np.load("actlist.npy")
# actonehot = to_categorical(action)
# np.save("newY",np.array(actonehot))
#
# print('Original label:\n', action[5:9])
# print('After conversion to one-hot:\n', actonehot[5:9])

""" Akhir Konversi """


""" Get Sample For Staking """

samplefile = "newdata/frame0.png";
x_t = imread(samplefile)
x_t = skimage.color.rgb2gray(x_t)
x_t = skimage.transform.resize(x_t,(80,80))
x_t = skimage.exposure.rescale_intensity(x_t,out_range=(0,255))
x_t = x_t / 255.0
s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
#s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  #1*80*80*4

print('Size Before Staking:\n', x_t.shape)
print('Size After  Staking :\n', s_t.shape)

""" End For Sample Staking """


""" Preparing Data """
a = 90001
b = 100000
sizenya = str(a)+"-"+str(b)
DataX = []
for i in range(a, b):
    print("Data Ke : ", i)
    foldername = "newdata/";
    filename = foldername+"frame"+str(i)+".png"
    img = imread(filename)
    x_t1 = skimage.color.rgb2gray(img)
    x_t1 = skimage.transform.resize(x_t1,(80,80))
    x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))
    x_t1 = x_t1 / 255.0
    x_t1 = x_t1.reshape(x_t1.shape[0], x_t1.shape[1], 1) #1x80x80x1
    s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)
    DataX.append(s_t1)
np.save("newX"+sizenya,np.array(DataX))

""" End For Preparing Data """
