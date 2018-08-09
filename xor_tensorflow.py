import tensorflow as tf
import numpy as np
import random as rand

nb_train = 64
data = []
labels = []

for i in range(nb_train):
    x = rand.randint(0, 1)
    y = rand.randint(0, 1)
    noisex = np.random.normal()*(10**-1)
    noisey = np.random.normal()*(10**-1)
    labels.append(1) if x != y else labels.append(0)
    data.append([x+noisex, y+noisey])
    
x_train = np.array(data)
y_train = np.array(labels)

x_test = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]])

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.int32)
# building network
# 2 in, 2 out (2 hidden layers)
weights1 = tf.Variable(tf.truncated_normal((2,2)))

# 2 in (from hidden layers, 2 output 0/1)
weights2 = tf.Variable(tf.truncated_normal((2,2)))

bias1 = tf.Variable(tf.zeros(2))
bias2 = tf.Variable(tf.zeros(2))

one_hot = tf.one_hot(y, 2)

hidden_layer = tf.tanh(tf.add(tf.matmul(x, weights1), bias1))
output_predictions = tf.tanh(tf.add(tf.matmul(hidden_layer, weights2), bias2))
loss = tf.losses.mean_squared_error(one_hot, output_predictions)

learning_rate = 0.1
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

nb_epoch = 10000
test_score = None
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(nb_epoch):
        _, l = sess.run([optimizer, loss], feed_dict={x: x_train, y: y_train})
        if not i % 1000:
            print(l)
    test_score = sess.run(output_predictions, feed_dict={x: x_test})

print(test_score)