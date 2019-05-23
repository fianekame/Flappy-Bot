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
FINAL_EPSILON = 0.0001 # final value of epsilon
FRAME_PER_ACTION = 1
ACTIONS = 2 # number of valid actions


def playNetwork(model, sess):
    # open up a game state to communicate with emulator
    game_state = game.GameState()
    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = skimage.color.rgb2gray(x_t)
    x_t = skimage.transform.resize(x_t,(80,80))
    x_t = skimage.exposure.rescale_intensity(x_t,out_range=(0,255))
    x_t = x_t / 255.0
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  #1*80*80*4

    t = 0
    epsilon = FINAL_EPSILON
    for i in range(0,5000):
    # while (True):
        r_t = 0
        action_index = 0
        a_t = np.zeros([ACTIONS])
        if t % FRAME_PER_ACTION == 0:
            q = model.predict(s_t)
            max_Q = np.argmax(q)

            action_index = max_Q
            a_t[max_Q] = 1

            # myrand = random.random();
            # print(myrand)
            # if myrand <= epsilon:
            #     print("----------Random Action----------")
            #     action_index = random.randrange(ACTIONS)
            #     a_t[action_index] = 1
            # else:

        #run the selected action and observed next state and reward
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        x_t1 = skimage.color.rgb2gray(x_t1_colored)
        x_t1 = skimage.transform.resize(x_t1,(80,80))
        x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))
        x_t1 = x_t1 / 255.0
        x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1) #1x80x80x1
        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)
        s_t = s_t1
        t = t + 1

        print("TIMESTEP", t,"/ ACTION", action_index, "/ Reward", r_t,"/ Terminal", terminal)

    print("Episode finished!")
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
