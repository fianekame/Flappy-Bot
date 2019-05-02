import numpy as np
import tensorflow as tf
import matplotlib as pyplot

""" CNN Architecture Following VGG-Net"""

def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x,W,strides=[1, strides, strides, 1],padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x,ksize=[1, k, k, 1],strides=[1, k, k, 1],padding='SAME')

def conv_net(x, weights, biases):
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, k=2)
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2)
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    conv3 = maxpool2d(conv3, k=2)
    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'])
    conv4 = maxpool2d(conv4, k=2)
    conv5 = conv2d(conv4, weights['wc5'], biases['bc5'])
    conv5 = maxpool2d(conv5, k=2)
    fc1 = tf.reshape(conv5, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Output, class prediction
    # finally we multiply the fully connected layer with the weights and add a bias term.
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

""" Inisialisasi """

training_iters = 200
learning_rate = 0.0001
batch_size = 128
n_input = 80
n_classes = 2

""" Get Data To Try"""

data_x = np.load('initial_data/DataX.npy')
data_y = np.load('initial_data/DataY.npy')
train_X = data_x[:500]
train_y = data_y[:500]
test_X = data_x[500:600]
test_y = data_y[500:600]

""" CNN Architecture Following VGG-Net"""

x = tf.placeholder("float", [None, 80,80,4])
y = tf.placeholder("float", [None, n_classes])
weights = {
    'wc1': tf.get_variable('W0', shape=(3,3,4,64), initializer=tf.contrib.layers.xavier_initializer()),
    'wc2': tf.get_variable('W1', shape=(3,3,64,128), initializer=tf.contrib.layers.xavier_initializer()),
    'wc3': tf.get_variable('W2', shape=(3,3,128,256), initializer=tf.contrib.layers.xavier_initializer()),
    'wc4': tf.get_variable('W3', shape=(3,3,256,512), initializer=tf.contrib.layers.xavier_initializer()),
    'wc5': tf.get_variable('W4', shape=(3,3,512,512), initializer=tf.contrib.layers.xavier_initializer()),
    'wd1': tf.get_variable('W5', shape=(3*3*512,512), initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('W6', shape=(512,n_classes), initializer=tf.contrib.layers.xavier_initializer()),
}
biases = {
    'bc1': tf.get_variable('B0', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'bc2': tf.get_variable('B1', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'bc3': tf.get_variable('B2', shape=(256), initializer=tf.contrib.layers.xavier_initializer()),
    'bc4': tf.get_variable('B3', shape=(512), initializer=tf.contrib.layers.xavier_initializer()),
    'bc5': tf.get_variable('B4', shape=(512), initializer=tf.contrib.layers.xavier_initializer()),
    'bd1': tf.get_variable('B5', shape=(512), initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('B6', shape=(2), initializer=tf.contrib.layers.xavier_initializer()),
}

pred = conv_net(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    summary_writer = tf.summary.FileWriter('./Output', sess.graph)
    saver = tf.train.Saver(max_to_keep=1)
    for i in range(training_iters):
        for batch in range(len(train_X)//batch_size):
            batch_x = train_X[batch*batch_size:min((batch+1)*batch_size,len(train_X))]
            batch_y = train_y[batch*batch_size:min((batch+1)*batch_size,len(train_y))]
            # Calculate batch loss and accuracy
            opt = sess.run(optimizer, feed_dict={x: batch_x,y: batch_y})
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,y: batch_y})
        print("Iter " + str(i) + ", Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
        print("Optimization Finished!")
        test_acc,valid_loss = sess.run([accuracy,cost], feed_dict={x: test_X,y : test_y})
        train_loss.append(loss)
        test_loss.append(valid_loss)
        train_accuracy.append(acc)
        test_accuracy.append(test_acc)
        print("Testing Accuracy:","{:.5f}".format(test_acc))

    saver.save(sess,'saved_networks/savedmodel.ckpt')
    summary_writer.close()
