import tensorflow as tf
import Read_left_15,Read_front_32
import numpy as np

batch_size = 2

xs = tf.placeholder(tf.float32, [224, 224])   # 224x224
ys = tf.placeholder(tf.float32, [32, 32])  # ground truth

image_ff, label_ff = Read_front_32.read_and_decode('front_face_32.tfrecords')
print(image_ff)
'''
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
coord=tf.train.Coordinator()
threads= tf.train.start_queue_runners(coord=coord)
'''
np_example_label_ff = np.zeros((batch_size, 1))
np_example_ff = np.zeros((batch_size, 32, 32))
with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(batch_size):
        np_example_ff[i], np_example_label_ff[i] = sess.run([image_ff, label_ff])  # 在会话中取出image
        print(np_example_ff[i])
    coord.request_stop()
    coord.join(threads)