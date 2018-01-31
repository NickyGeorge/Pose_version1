from __future__ import print_function
import tensorflow as tf
import numpy as np
import Read_left_15,Read_front_32

batch_size = 32
learing_rate = 0.01

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

def fully_connect(x, W):
    return tf.nn.conv2d(x, W, strides=[], padding='')

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [batch_size, 224, 224, 1])   # 224x224
ys = tf.placeholder(tf.float32, [batch_size, 32, 32])  # ground truth
keep_prob = tf.placeholder(tf.float32)
y_gt = tf.reshape(ys, [-1, 1024])  # [batch_size, 1024]

##-----------conv_1_block layer---------- ##
## conv1_1 layer
W_conv1_1 = weight_variable([3,3, 1,64]) # patch 3x3, in size 1, out size 64
b_conv1_1 = bias_variable([64])
h_conv1_1 = tf.nn.relu(conv2d(xs, W_conv1_1) + b_conv1_1) # output size 224x224x64
## conv1_2 layer
W_conv1_2 = weight_variable([3, 3, 64,64]) # patch 3x3, in size 64, out size 64
b_conv1_2 = bias_variable([64])
h_conv1_2 = tf.nn.relu(conv2d(h_conv1_1, W_conv1_2) + b_conv1_2) # output size 224x224x64
conv_1_pool = max_pool_2x2(h_conv1_2) # output size 112x112x64


##----------conv_2 block_layer---------- ##
## conv2_1 layer
W_conv2_1 = weight_variable([3, 3, 64, 128]) # patch 3x3, in size 64, out size 128
b_conv2_1 = bias_variable([128])
h_conv2_1 = tf.nn.relu(conv2d(conv_1_pool, W_conv2_1) + b_conv2_1) # output size 112x112x128
## conv2_2 layer
W_conv2_2 = weight_variable([3, 3, 128, 128]) # patch 3x3, in size 128, out size 128
b_conv2_2 = bias_variable([128])
h_conv2_2 = tf.nn.relu(conv2d(h_conv2_1, W_conv2_2) + b_conv2_2) # output size 112x112x128
conv_2_pool = max_pool_2x2(h_conv2_2) # output size 56x56x128


##----------conv_3 block_layer---------- ##
## conv3_1 layer
W_conv3_1 = weight_variable([3, 3, 128, 256]) # kernel_size 3x3, in channel 128, out channel 256
b_conv3_1 = bias_variable([256])
h_conv3_1 = tf.nn.relu(conv2d(conv_2_pool, W_conv3_1) + b_conv3_1) # output_size 56x56x256
## conv3_2 layer
W_conv3_2 = weight_variable([3, 3, 256, 256])
b_conv3_2 = bias_variable([256])
h_conv3_2 = tf.nn.relu(conv2d(h_conv3_1, W_conv3_2) + b_conv3_2) # output_size 56x56x256
## conv3_3 layer
W_conv3_3 = weight_variable([3, 3, 256, 256])
b_conv3_3 = bias_variable([256])
h_conv3_3 = tf.nn.relu(conv2d(h_conv3_2, W_conv3_3) + b_conv3_3) # output_size 56x56x256
conv_3_pool = max_pool_2x2(h_conv3_3) # output_size 28x28x256


##----------conv_4 block_layer---------- ##
## conv4_1 layer
W_conv4_1 = weight_variable([3, 3, 256, 512])
b_conv4_1 = bias_variable([512])
h_conv4_1 = tf.nn.relu(conv2d(conv_3_pool, W_conv4_1) + b_conv4_1) # output_size 28x28x512
## conv4_2 layer
W_conv4_2 = weight_variable([3, 3, 512, 512])
b_conv4_2 = bias_variable([512])
h_conv4_2 = tf.nn.relu(conv2d(h_conv4_1, W_conv4_2) + b_conv4_2)
## conv4_ layer
W_conv4_3 = weight_variable([3, 3, 512, 512])
b_conv4_3 = bias_variable([512])
h_conv4_3 = tf.nn.relu(conv2d(h_conv4_2, W_conv4_3) + b_conv4_3)
conv_4_pool = max_pool_2x2(h_conv4_3) # output_size 14x14x512


##----------conv_5 block_layer---------- ##
## conv5_1 layer
W_conv5_1 = weight_variable([3, 3, 512, 512])
b_conv5_1 = bias_variable([512])
h_conv5_1 = tf.nn.relu(conv2d(conv_4_pool, W_conv5_1) + b_conv5_1) ##  output_size 14x14x512
## conv5_2 layer
W_conv5_2 = weight_variable([3, 3, 512, 512])
b_conv5_2 = bias_variable([512])
h_conv5_2 = tf.nn.relu(conv2d(h_conv5_1, W_conv5_2) + b_conv5_2) ##  output_size 14x14x512
## conv5_3 layer
W_conv5_3 = weight_variable([3, 3, 512, 512])
b_conv5_3 = bias_variable([512])
h_conv5_3 = tf.nn.relu(conv2d(h_conv5_2, W_conv5_3) + b_conv5_3)
conv_5_pool = max_pool_2x2(h_conv5_3)   ##  output_size 7x7x512
conv_5_pool_flat = tf.reshape(conv_5_pool, [-1, 7*7*512]) # samplesx25088


## fcn_1 layer
W_fcn1 = weight_variable([7*7*512, 4096])
b_fcn1 = bias_variable([4096])
h_fcn1 = tf.nn.relu(tf.matmul(conv_5_pool_flat, W_fcn1) + b_fcn1) # output_size 4096

## fcn_2 layer
W_fcn2 = weight_variable([4096, 4096])
b_fcn2 = bias_variable([4096])
h_fcn2 = tf.nn.relu(tf.matmul(h_fcn1, W_fcn2) + b_fcn2) # output_size 4096

## prediction layer
W_fcn3 = weight_variable([4096, 1024])
b_fcn3 = bias_variable([1024])
prediction = tf.nn.relu(tf.matmul(h_fcn2, W_fcn3) + b_fcn3) # output_size 1024

## the error between prediction and real data
euclidean = tf.sqrt(tf.reduce_sum(tf.square(prediction - y_gt), reduction_indices=[1]))   # 欧式距离
train_step = tf.train.GradientDescentOptimizer(learing_rate).minimize(euclidean)

## train
image_lt_15, label_lt_15 = Read_left_15.read_and_decode("left_15.tfrecords")
image_ff, label_ff = Read_front_32.read_and_decode("front_face_32.tfrecords")
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
coord=tf.train.Coordinator()
threads= tf.train.start_queue_runners(coord=coord)
np_example_lt_15 = np.zeros((batch_size,224,224))
np_l_lt_15= np.zeros((batch_size,1))

np_example_ff = np.zeros((batch_size,32,32))
np_l_ff = np.zeros((batch_size,1))

with tf.Session() as sess:
    try:
        sess.run(tf.initialize_all_variables
                 ())
        for iteration in range(60):
            for i in range(batch_size):
                print('start get image and lables...')
                np_example_lt_15[i], np_l_lt_15[i] = sess.run([image_lt_15, label_lt_15])  # 在会话中取出image和label
                np_example_ff[i], np_l_ff[i] = sess.run([image_ff, label_ff])
                print('start train_step...')
            sess.run(train_step, feed_dict={xs: np_example_lt_15, ys: np_example_ff})
            print('train_step over')
            print('the '+ iteration + 'th' + 'iteration euclidean is below: ')
            print(sess.run(euclidean, feed_dict={xs: np_example_lt_15, ys: np_example_ff}))
    except tf.errors.OutOfRangeError:
        print('done!')
    finally:
        coord.request_stop()
        coord.join(threads)

