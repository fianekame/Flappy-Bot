import numpy as np
import tensorflow as tf
import matplotlib as pyplot

""" Get Data To Try"""

data_x = np.load('initial_data/DataX.npy')
data_y = np.load('initial_data/DataY.npy')
test_X = data_x[600:700]
test_y = data_y[600:700]


v1 = tf.get_variable("v1", shape=[3])


saver = tf.train.Saver()
sess = tf.Session()

with tf.Session() as sess:
  sess.run(init_op)
  # Save the variables to disk.
  save_path = saver.save(sess, "/saved_networks/savedmodel.ckpt")
  print("Model saved in path: %s" % save_path)
