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

_ROOTPATH = "/home/galgadot/Documents/Skripsi/FlappyBot/"
GAME = 'bird' # the name of the game being played for log files
ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 100000. # timesteps to observe before training
EXPLORE = 2000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.0001 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 64 # size of minibatch
FRAME_PER_ACTION = 1


def convertbos(img):
    img = img.astype('float32')
    img = img / 255.
    img = img.reshape(-1,80,80,4)
    return img

def playNetwork(loaded_model, sess):
    game_state = game.GameState()
    # 10 = 0 , 01 = 1
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)



    t = 0
    epsilon = INITIAL_EPSILON
    for gbr in range(0,1000):
        # choose an action epsilon greedily
        readout_t = loaded_model.predict(convertbos(s_t))
        a_t = np.zeros([ACTIONS])
        action_index = 0
        if t % FRAME_PER_ACTION == 0:
            action_index = np.argmax(readout_t)
            a_t[action_index] = 1
            # if random.random() <= epsilon:
            #     print("----------Random Action----------")
            #     action_index = random.randrange(ACTIONS)
            #     a_t[random.randrange(ACTIONS)] = 1
            # else:
            #     action_index = np.argmax(readout_t)
            #     a_t[action_index] = 1
        else:
            a_t[0] = 1 # do nothing

        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)
        # x_t1 = x_t1.astype('float32')
        # x_t1 = x_t1 / 255.
        # print(s_t.shape)
        # # x_t1 = np.transpose(x_t1)
        # print(x_t1.shape)
        # # print(x_t1.shape==s_t.shape)
        # print(s_t)
        # print(x_t1)
        # np.save('s_t.npy', s_t)
        # np.save('x_t1.npy', x_t1)
        s_t = s_t1
        t += 1

        print("TIMESTEP", t, "/ EPSILON", epsilon, "/ ACTION", action_index, "/ R_MAX %e" % np.max(readout_t))

def playGame():
    sess = tf.InteractiveSession()
    json_file = open(_ROOTPATH+'saved_networks/saved_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(_ROOTPATH+"saved_networks/saved_model.h5")
    print("Loaded model from disk")
    playNetwork(loaded_model,sess)

def main():
    playGame()

if __name__ == "__main__":
    main()
