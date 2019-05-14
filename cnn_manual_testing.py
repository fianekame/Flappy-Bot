import numpy as np
import tensorflow as tf
import matplotlib as pyplot

""" Get Data To Try"""

data_x = np.load('initial_data/DataX.npy')
data_y = np.load('initial_data/DataY.npy')
test_x = data_x[600:700]
test_y = data_y[600:700]

# # saver = tf.train.Saver()
# sess = tf.Session()
#
model_path = "saved_networks/savedmodel.ckpt"
# # inference_graph = tf.Graph()
# # inference_graph = tf.get_default_graph()
#
# output = inference_graph.get_tensor_by_name("output:0")
# x = inference_graph.get_tensor_by_name("x:0")
#
with tf.Session() as sess:
    loader = tf.train.import_meta_graph(model_path+".meta")
    loader.restore(sess,model_path)
    graph = tf.get_default_graph()
    pred = graph.get_tensor_by_name('output:0')
    x = graph.get_tensor_by_name('x:0')
    output_class = sess.run(pred, feed_dict={x:test_x})
    predicted_classes = np.argmax(np.round(output_class),axis=1)
    print(output_class)
    print(predicted_classes)
    print(np.argmax(np.round(test_y),axis=1))
