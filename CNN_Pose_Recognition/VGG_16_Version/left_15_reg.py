from __future__ import print_function
import tensorflow as tf

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 50176])   # 28x28
ys = tf.placeholder(tf.float32, [None, 1024])  #output
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 224, 224, 1])
# print(x_image.shape)  # [n_samples, 28,28,1]

##-----------conv_1_block layer---------- ##
## conv1_1 layer
W_conv1_1 = weight_variable([3,3, 1,64]) # patch 3x3, in size 1, out size 32
b_conv1_1 = bias_variable([64])
h_conv1_1 = tf.nn.relu(conv2d(x_image, W_conv1_1) + b_conv1_1) # output size 224x224x64
## conv1_2 layer
W_conv1_2 = weight_variable([3, 3, 64,64]) # patch 3x3, in size 64, out size 64
b_conv1_2 = bias_variable([64])
h_conv1_2 = tf.nn.relu(conv2d(h_conv1_1, W_conv1_2) + b_conv1_2) # output size 224x224x64
conv1_pool = max_pool_2x2(h_conv1_2) # output size 112x112x64
##----------conv_2 block_layer---------- ##
## conv2_1 layer
W_conv2_1 = weight_variable([112, 112, 64, 128]) # patch 3x3, in size 64, out size 128
b_conv2_1 = bias_variable([128])
h_conv2_1 = tf.nn.relu(conv2d(conv1_pool, W_conv2_1) + b_conv2_1) # output size 112x112x128
## conv2_2 layer
W_conv2_2 = weight_variable([112, 112, 128, 128]) # patch 3x3, in size 128, out size 128
b_conv2_2 = bias_variable([128])
h_conv2_2 = tf.nn.relu(conv2d([h_conv2_1, W_conv2_1]) + b_conv2_2) # output size 112x112x128
conv2_pool = max_pool_2x2(h_conv2_2) # output size 56x56x128
##----------conv_3 block_layer---------- ##

## fc1 layer ##
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## fc2 layer ##
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()
# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 50 == 0:
        print(compute_accuracy(
            mnist.test.images[:1000], mnist.test.labels[:1000]))