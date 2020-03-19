import tensorflow.compat.v1 as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from tensorflow_core.examples.tutorials.mnist import input_data

tf.compat.v1.disable_eager_execution()
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# print(mnist.train.images.max())
x = mnist.train.images
xph = tf.placeholder(tf.float32, [None, 784])
yph = tf.placeholder(tf.float32, [None, 10])
# [55000,784][784,60]
w1 = tf.Variable(tf.truncated_normal([784, 60]))
b1 = tf.Variable(tf.truncated_normal([60]))
w2 = tf.Variable(tf.truncated_normal([60, 60]))
b2 = tf.Variable(tf.truncated_normal([60]))
w3 = tf.Variable(tf.truncated_normal([60, 10]))
b3 = tf.Variable(tf.truncated_normal([10]))


y = tf.nn.relu(xph @ w1 + b1)
y = tf.nn.relu(y @ w2 + b2)
y = y @ w3 + b3

y_true = tf.placeholder(tf.float32, shape=[None, 10])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=yph, logits=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cross_entropy)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(100):
        batch_x, batch_y = mnist.train.next_batch(150)
        sess.run(train, feed_dict={xph: batch_x, yph: batch_y})
    matches = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))
    acc = tf.reduce_mean(tf.cast(matches, tf.float32))
    print(sess.run(acc, feed_dict={xph: mnist.test.images, y_true: mnist.test.labels}))
    w_1, b_1, w_2, b_2, w_3, b_3 = sess.run([w1, b1, w2, b2, w3, b3])

