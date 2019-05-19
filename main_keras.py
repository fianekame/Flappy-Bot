#!/usr/bin/env python
from __future__ import print_function

import tensorflow as tf
import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
import keras
from keras.models import model_from_json
from keras.initializers import normal, identity
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam
import tensorflow as tf
from skimage import data, io
from skimage.io import imsave, imread

import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer


_ROOTPATH = "/home/galgadot/Documents/Skripsi/FlappyBot/"
GAME = 'bird' # the name of the game being played for log files
CONFIG = 'nothreshold'
ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVATION = 3200. # timesteps to observe before training
EXPLORE = 3000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-4


def playNetwork(model, sess):
    game_state = game.GameState()
    # 10 = 0 , 01 = 1
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t2 = x_t
    x_t2 = cv2.cvtColor(cv2.resize(x_t2, (80, 80)), cv2.COLOR_BGR2GRAY)
    x_t = skimage.color.rgb2gray(x_t)
    x_t = skimage.transform.resize(x_t,(80,80))
    x_t = skimage.exposure.rescale_intensity(x_t,out_range=(0,255))

    ret, x_t2 = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    x_t2 = x_t2 / 255.0
    s_t2 = np.stack((x_t2, x_t2, x_t2, x_t2), axis=2)
    x_t = x_t / 255.0
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
    #print (s_t.shape)

    #In Keras, need to reshape
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  #1*80*80*4

    t = 0
    epsilon = FINAL_EPSILON
    for gbr in range(0,5000):
        # choose an action epsilon greedily
        action_index = 0
        a_t = np.zeros([ACTIONS])
        #choose an action epsilon greedy
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                a_t[action_index] = 1
            else:
                q = model.predict(s_t)       #input a stack of 4 images, get the prediction
                max_Q = np.argmax(q)
                action_index = max_Q
                a_t[max_Q] = 1

        # #We reduced the epsilon gradually
        # if epsilon > FINAL_EPSILON and t > OBSERVE:
        #     epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        #run the selected action and observed next state and reward
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        x_t1 = skimage.color.rgb2gray(x_t1_colored)
        # imsave("test1.png", x_t1)
        x_t1 = skimage.transform.resize(x_t1,(80,80))
        # imsave("test2.png", x_t1)
        x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))
        # imsave("test3.png", x_t1)
        x_t1 = x_t1 / 255.0
        x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1) #1x80x80x1
        # print(x_t1.shape)
        # print(s_t.shape)
        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)
        s_t = s_t1
        t += 1
        print("TIMESTEP", t, "/ EPSILON", epsilon, "/ ACTION", action_index)

def buildmodel():
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

    adam = Adam(lr=1e-4)
    model.compile(loss='mse',optimizer=adam)
    print("We finish building the model")
    return model

def playGame():
    sess = tf.InteractiveSession()
    model = buildmodel()
    print ("Now we load weight")
    model.load_weights("saved_networks/model.h5")
    adam = Adam(lr=1e-4)
    model.compile(loss='mse',optimizer=adam)
    print ("Weight load successfully")
    playNetwork(model,sess)

def main():
    playGame()

if __name__ == "__main__":
    main()
