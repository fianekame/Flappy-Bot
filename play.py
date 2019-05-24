# 1248173607
import sys
sys.path.append("utils/game/")
import wrapped_flappy_bird as game
import tensorflow as tf
import skimage as skimage
import random
import numpy as np
import keras

from keras.models import model_from_json
from keras.optimizers import SGD , Adam
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer
from skimage.io import imsave, imread, imshow

_ROOTPATH = "/home/galgadot/Documents/Skripsi/FlappyBot/"
ACTIONS = 2 # number of valid actions


def playNetwork(model, sess):
    game_state = game.GameState()
    action = np.zeros(ACTIONS)
    action[0] = 1
    img, reward = game_state.frame_step(action)
    img = skimage.color.rgb2gray(img)
    img = skimage.transform.resize(img,(80,80))
    img = skimage.exposure.rescale_intensity(img,out_range=(0,255))
    img = img / 255.0
    stack_img = np.stack((img, img, img, img), axis=2)
    stack_img = stack_img.reshape(1, stack_img.shape[0], stack_img.shape[1], stack_img.shape[2])  #1*80*80*4

    frame = 0
    for i in range(0,500):
    # while (True):
        reward = 0
        action_index = 0
        action = np.zeros([ACTIONS])
        predict = model.predict(stack_img)
        result = np.argmax(predict)
        action_index = result
        action[result] = 1

        #run the selected action and observed next state and reward
        img, reward = game_state.frame_step(action)
        img = skimage.color.rgb2gray(img)
        img = skimage.transform.resize(img,(80,80))
        img = skimage.exposure.rescale_intensity(img, out_range=(0, 255))
        img = img / 255.0
        img = img.reshape(1, img.shape[0], img.shape[1], 1) #1x80x80x1
        stack = np.append(img, stack_img[:, :, :, :3], axis=3)
        stack_img = stack
        frame = frame + 1

        print("Frame", frame,"/ Action", action_index, "/ Reward", reward)

    print("End")
    print("************************")


def playGame():
    sess = tf.InteractiveSession()

    # load json and create model
    json_file = open('saved_networks/saved_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("saved_networks/saved_model.h5")
    adam = Adam(lr=1e-4)
    loaded_model.compile(loss='mse',optimizer=adam)
    print("Loaded model from disk")
    playNetwork(loaded_model,sess)

def main():
    playGame()

if __name__ == "__main__":
    main()
