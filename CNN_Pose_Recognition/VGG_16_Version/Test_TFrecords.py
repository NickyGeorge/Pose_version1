import tensorflow as tf
import Read_left_15,Read_front_32
import  numpy as np

batch_size = 2

xs = tf.placeholder(tf.float32, [224, 224])   # 224x224
ys = tf.placeholder(tf.float32, [32, 32])  # ground truth

image_lt_15, label_lt_15 = Read_left_15.read_and_decode("left_15.tfrecords")
image_ff, label_ff = Read_front_32.read_and_decode("front_face_32.tfrecords")
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
coord=tf.train.Coordinator()
threads= tf.train.start_queue_runners(coord=coord)
print(image_lt_15)
print(image_ff)

np_example_lt_15 = np.zeros((batch_size,224,224))
np_l_lt_15= np.zeros((batch_size,1))

np_example_ff = np.zeros((batch_size,32,32))
np_l_ff = np.zeros((batch_size,1))

with tf.Session() as sess:
        for i in range(batch_size):
            np_example_lt_15[i] = sess.run([image_lt_15])  # 在会话中取出image和label

        print(sess.run(np_example_lt_15))

