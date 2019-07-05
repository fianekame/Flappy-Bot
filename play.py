# 1248173607
# Last Commit Sebelum Sidang,
import sys
sys.path.append("utils/game/")
import wrapped_flappy_bird as game
import tensorflow as tf
import skimage as skimage
import random
import numpy as np
import keras
import datetime
import time
import json


from keras.models import model_from_json
from keras.optimizers import SGD , Adam
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer
from skimage.io import imsave, imread, imshow


_ROOTPATH = "/home/galgadot/Documents/Skripsi/FlappyBot/"
ACTIONS = 2

def loadFromFile():
    with open('save_data.json', 'r') as fp:
        data = json.load(fp)
    return data['hgscore'], data['hgtime']

def saveToFile(hgscore,hgtime):

    data =	{
      "hgscore": hgscore,
      "hgtime": hgtime,
      "lastsaved": str(datetime.datetime.now())
    }
    with open('save_data.json', 'w') as fp:
        json.dump(data, fp)

def playNetwork(model, sess):
    notDead = True
    hgscore, hgtime = loadFromFile()
    game_state = game.GameState()
    readytoplay = game_state.showWelcomeAnimation()
    if readytoplay:
        import time
        skor = 0
        start = time.time()
        action = np.zeros(ACTIONS)
        action[0] = 1
        img, reward, crashinfo = game_state.frame_step(action)
        img = skimage.color.rgb2gray(img)
        img = skimage.transform.resize(img,(80,80))
        img = skimage.exposure.rescale_intensity(img,out_range=(0,255))
        img = img / 255.0
        stack_img = np.stack((img, img, img, img), axis=2)
        stack_img = stack_img.reshape(1, stack_img.shape[0], stack_img.shape[1], stack_img.shape[2])  #1*80*80*4
        frame = 0
        # for i in range(0,1000):
        while (True):
            if notDead:
                reward = 0
                action_index = 0
                action = np.zeros([ACTIONS])
                predict = model.predict(stack_img)
                result = np.argmax(predict)
                action_index = result
                action[result] = 1
                # action = [0,1] todo cek crash info
                actionstatus = "Flap" if action_index == 1 else "Not-Flap"
                img, reward, crashinfo = game_state.frame_step(action)
                img = skimage.color.rgb2gray(img)
                img = skimage.transform.resize(img,(80,80))
                img = skimage.exposure.rescale_intensity(img, out_range=(0, 255))
                img = img / 255.0
                img = img.reshape(1, img.shape[0], img.shape[1], 1)
                stack = np.append(img, stack_img[:, :, :, :3], axis=3)
                stack_img = stack
                frame = frame + 1
                if reward != -1:
                    if reward == 1:
                        skor = skor + reward
                    print("Hidup | Frame", frame,"/ Action", action_index,"|",actionstatus, "/ Reward", reward)
                else :
                    end = time.time()
                    print("Mati | Frame", frame, "/ Reward", reward)
                    notDead = False
            else :
                time_taken = int(end - start)
                timestr = str(datetime.timedelta(seconds=time_taken))
                crashinfo['timetake'] = timestr
                crashinfo['hgscore'] = hgscore
                crashinfo['hgtime'] = str(datetime.timedelta(seconds=hgtime))
                if skor > hgscore:
                    hgscore = skor
                if time_taken > hgtime:
                    hgtime = time_taken
                saveToFile(hgscore,hgtime)
                todo = game_state.showGameOverScreen(crashinfo)
                if todo:
                    skor = 0
                    crashinfo = {}
                    notDead = True

def loadModel():
    json_file = open('saved_networks/saved_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("saved_networks/saved_model.h5")
    adam = Adam(lr=1e-4)
    loaded_model.compile(loss='mse',optimizer=adam)
    print("Loaded model from disk")
    return loaded_model

def playGame():
    sess = tf.InteractiveSession()
    playNetwork(loadModel(),sess)

def main():
    playGame()

if __name__ == "__main__":
    main()
