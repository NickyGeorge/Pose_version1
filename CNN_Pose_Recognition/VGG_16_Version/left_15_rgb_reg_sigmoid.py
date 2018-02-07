from __future__ import print_function
import tensorflow as tf
import numpy as np
import front_face_rgb
import left_15_rgb
#import Read_left_15,Read_front_32
import skimage.io as io
from skimage import  color,transform

batch_size = 20
iteration = 1000
learing_rate = 0.001

def weight_variable(shape):
    initial = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
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
xs = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])   # 224x224x3
ys = tf.placeholder(tf.float32, [batch_size, 32, 32, 3])  # ground truth
keep_prob = tf.placeholder(tf.float32)
y_gt = tf.reshape(ys, [-1, 3072])  # [batch_size, 3072]
##-----------conv_1_block layer---------- ##
## conv1_1 layer
W_conv1_1 = weight_variable([3,3, 3,64]) # patch 3x3, in size 1, out size 64
b_conv1_1 = bias_variable([64])
h_conv1_1 = tf.nn.sigmoid(conv2d(xs, W_conv1_1) + b_conv1_1) # output size 224x224x64
## conv1_2 layer
W_conv1_2 = weight_variable([3, 3, 64,64]) # patch 3x3, in size 64, out size 64
b_conv1_2 = bias_variable([64])
h_conv1_2 = tf.nn.sigmoid(conv2d(h_conv1_1, W_conv1_2) + b_conv1_2) # output size 224x224x64
conv_1_pool = max_pool_2x2(h_conv1_2) # output size 112x112x64


##----------conv_2 block_layer---------- ##
## conv2_1 layer
W_conv2_1 = weight_variable([3, 3, 64, 128]) # patch 3x3, in size 64, out size 128
b_conv2_1 = bias_variable([128])
h_conv2_1 = tf.nn.sigmoid(conv2d(conv_1_pool, W_conv2_1) + b_conv2_1) # output size 112x112x128
## conv2_2 layer
W_conv2_2 = weight_variable([3, 3, 128, 128]) # patch 3x3, in size 128, out size 128
b_conv2_2 = bias_variable([128])
h_conv2_2 = tf.nn.sigmoid(conv2d(h_conv2_1, W_conv2_2) + b_conv2_2) # output size 112x112x128
conv_2_pool = max_pool_2x2(h_conv2_2) # output size 56x56x128


##----------conv_3 block_layer---------- ##
## conv3_1 layer
W_conv3_1 = weight_variable([3, 3, 128, 256]) # kernel_size 3x3, in channel 128, out channel 256
b_conv3_1 = bias_variable([256])
h_conv3_1 = tf.nn.sigmoid(conv2d(conv_2_pool, W_conv3_1) + b_conv3_1) # output_size 56x56x256
## conv3_2 layer
W_conv3_2 = weight_variable([3, 3, 256, 256])
b_conv3_2 = bias_variable([256])
h_conv3_2 = tf.nn.sigmoid(conv2d(h_conv3_1, W_conv3_2) + b_conv3_2) # output_size 56x56x256
## conv3_3 layer
W_conv3_3 = weight_variable([3, 3, 256, 256])
b_conv3_3 = bias_variable([256])
h_conv3_3 = tf.nn.sigmoid(conv2d(h_conv3_2, W_conv3_3) + b_conv3_3) # output_size 56x56x256
conv_3_pool = max_pool_2x2(h_conv3_3) # output_size 28x28x256


##----------conv_4 block_layer---------- ##
## conv4_1 layer
W_conv4_1 = weight_variable([3, 3, 256, 512])
b_conv4_1 = bias_variable([512])
h_conv4_1 = tf.nn.sigmoid(conv2d(conv_3_pool, W_conv4_1) + b_conv4_1) # output_size 28x28x512
## conv4_2 layer
W_conv4_2 = weight_variable([3, 3, 512, 512])
b_conv4_2 = bias_variable([512])
h_conv4_2 = tf.nn.sigmoid(conv2d(h_conv4_1, W_conv4_2) + b_conv4_2)
## conv4_ layer
W_conv4_3 = weight_variable([3, 3, 512, 512])
b_conv4_3 = bias_variable([512])
h_conv4_3 = tf.nn.sigmoid(conv2d(h_conv4_2, W_conv4_3) + b_conv4_3)
conv_4_pool = max_pool_2x2(h_conv4_3) # output_size 14x14x512


##----------conv_5 block_layer---------- ##
## conv5_1 layer
W_conv5_1 = weight_variable([3, 3, 512, 512])
b_conv5_1 = bias_variable([512])
h_conv5_1 = tf.nn.sigmoid(conv2d(conv_4_pool, W_conv5_1) + b_conv5_1) ##  output_size 14x14x512
## conv5_2 layer
W_conv5_2 = weight_variable([3, 3, 512, 512])
b_conv5_2 = bias_variable([512])
h_conv5_2 = tf.nn.sigmoid(conv2d(h_conv5_1, W_conv5_2) + b_conv5_2) ##  output_size 14x14x512
## conv5_3 layer
W_conv5_3 = weight_variable([3, 3, 512, 512])
b_conv5_3 = bias_variable([512])
h_conv5_3 = tf.nn.sigmoid(conv2d(h_conv5_2, W_conv5_3) + b_conv5_3)
conv_5_pool = max_pool_2x2(h_conv5_3)   ##  output_size 7x7x512
conv_5_pool_flat = tf.reshape(conv_5_pool, [-1, 7*7*512]) # samplesx25088


## fcn_1 layer
W_fcn1 = weight_variable([7*7*512, 4096])
b_fcn1 = bias_variable([4096])
h_fcn1 = tf.nn.sigmoid(tf.matmul(conv_5_pool_flat, W_fcn1) + b_fcn1) # output_size 4096

## fcn_2 layer
W_fcn2 = weight_variable([4096, 4096])
b_fcn2 = bias_variable([4096])
h_fcn2 = tf.nn.sigmoid(tf.matmul(h_fcn1, W_fcn2) + b_fcn2) # output_size 4096

## prediction layer
W_fcn3 = weight_variable([4096, 32*32*3])
b_fcn3 = bias_variable([32*32*3])
prediction = tf.nn.sigmoid(tf.matmul(h_fcn2, W_fcn3) + b_fcn3) # output_size 32*32*3

## the error between prediction and real data
euclidean = tf.sqrt(tf.reduce_sum(tf.square(prediction - y_gt), reduction_indices=[1]))   # 欧式距离
train_step = tf.train.GradientDescentOptimizer(learing_rate).minimize(euclidean)

## train
image_ff = np.zeros((200, 32, 32, 3))
image_lt_15 = np.zeros((200, 224, 224, 3))
image_lt_15 = left_15_rgb.Collect_Pic()
image_ff = front_face_rgb.Collect_Pic()
#print(image_lt_15[8])
#print(image_ff[8])
#print(len(image_ff))
#print(len(image_lt_15))
#print(image_lt_15.shape)
#print(image_ff.shape)
np_example_lt_15 = np.zeros((batch_size,224,224,3))
np_example_ff = np.zeros((batch_size,32,32,3))

def Normalize_results(x):
    euclidean = 0
    for i in range(len(x)):
        euclidean += x[i]
    return euclidean/len(x)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    k = 0
    results_prediction = np.zeros((iteration ,batch_size, 32*32*3))
    results_y_gt = np.zeros((iteration, batch_size, 32*32*3))
    results_1 = np.zeros((iteration,1))
    for j in range(iteration):
        for i in range(batch_size):
            if k >= 200:
                k = 0
            #print(image_lt_15[j])
            np_example_lt_15[i] = image_lt_15[k]
            np_example_ff[i] = image_ff[k]
            k = k + 1
        #print(np_example_lt_15)
        #print(np_example_ff)
        #print('------------------------------')

        print('the '+ str(j) + 'th' + ' iteration euclidean is below: ')
        print(Normalize_results(sess.run(euclidean, feed_dict={xs: np_example_lt_15, ys: np_example_ff})))
        results_y_gt[j] = sess.run(y_gt, feed_dict={xs: np_example_lt_15, ys: np_example_ff})
        results_prediction[j] = sess.run(prediction, feed_dict={xs: np_example_lt_15, ys: np_example_ff})
        results_1[j] = Normalize_results(sess.run(euclidean, feed_dict={xs: np_example_lt_15, ys: np_example_ff}))
        sess.run(train_step, feed_dict={xs: np_example_lt_15, ys: np_example_ff})

with open('.\Results\left_15_rgb_reg_sigmoid\letf_15_reg_prediction_results.txt', 'w') as outfile:
    outfile.write('# Array shape: {0}\n'.format(results_prediction.shape))
    for data_slice in results_prediction:
        np.savetxt(outfile, data_slice, fmt='%-7.4f')
        outfile.write('# New slice\n')

with open('.\Results\left_15_rgb_reg_sigmoid\letf_15_reg_y_gt_results.txt', 'w') as outfile:
    outfile.write('# Array shape: {0}\n'.format(results_y_gt.shape))
    for data_slice in results_y_gt:
        np.savetxt(outfile, data_slice, fmt='%-7.4f')
        outfile.write('# New slice\n')

np.savetxt('.\Results\left_15_rgb_reg_sigmoid\left_15_reg_euclidean_results.txt', results_1)