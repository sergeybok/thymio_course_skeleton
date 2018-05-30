import tensorflow as tf
import numpy as np

from Data import get_data
import os, cv2, re

n_steps = 1000
lr = 0.0004
b_size = 40

stack = 1

max_t = 6
rnn_hid = 16

def rnn_batchdata(dx,dy,mt,bs,b,h,w,c):
    # dx: dataset_x, dy: dataset_y
    # mt: num_time_steps
    # bs: batch_size, b: step/batch
    # h: input height, w: input width, c: num_channels
    bx = np.empty((mt,bs,h,w,c))
    by = np.empty((mt,bs,1))
    b = b+mt*bs % dx.shape[0] 
    for i in range(bs):
        bx[:, i, :, :] = dx[b+(i*mt):b+(i*mt)+mt]
        by[:,i] = dy[b+(i*mt):b+(i*mt)+mt]

    return bx, by



def CNN(x):
    #small_x = tf.image.resize_images(x,[64,64])
    small_x = x
    conv1 = tf.layers.conv2d(small_x,filters=32,kernel_size=[7,7],padding='valid',activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(conv1,pool_size=[4,4],strides=[4,4])
    conv2 = tf.layers.conv2d(pool1, filters=8,kernel_size=[4,4],padding='valid',activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(conv2,pool_size=[3,3],strides=[3,3])
    #conv3 = tf.layers.conv2d(pool2,filters=8, kernel_size=[3,3],padding='valid')
    #pool3 = tf.layers.max_pooling2d(conv3,pool_size=[2,2],strides=[2,2])
    return pool2
    #return conv3



X = tf.placeholder(tf.float32,[None,84,84,stack])
Y = tf.placeholder(tf.float32,[None,1])

conv_out = CNN(X)

conv_flat = tf.contrib.layers.flatten(conv_out)
dense = tf.layers.dense(conv_flat, units=1)
prediction = tf.nn.tanh(dense)
Error = tf.losses.mean_squared_error(labels=Y,predictions=prediction)

optimizer = tf.train.AdamOptimizer(learning_rate=lr)
train_op = optimizer.minimize(Error)


# ------------RNN--------------------------------
Rx = tf.placeholder(tf.float32,[max_t,b_size,84,84,stack])
Ry = tf.placeholder(tf.float32,[max_t,b_size,1])

rnn_batch = tf.shape(Rx)[1]

rnn_cnn_input = tf.reshape(Rx,[max_t*rnn_batch,84,84,stack])

#print(rnn_cnn_input.get_shape().as_list())
rconv_out = CNN(rnn_cnn_input)
#print(rconv_out.get_shape().as_list())

rconv_shape = rconv_out.get_shape().as_list()
#rconv_shape = tf.shape(rconv_out)
rnn_conv_out = tf.reshape(rconv_out,[max_t,rnn_batch,rconv_shape[1],rconv_shape[2],rconv_shape[3]])
#rnn_conv_out = tf.reshape(rconv_out,[max_t,rnn_batch,-1])
#print(rnn_conv_out.get_shape().as_list())
rconv_flat = tf.reshape(rconv_out,[max_t,rnn_batch,rconv_shape[1]*rconv_shape[2]*rconv_shape[3]])

lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=rnn_hid,forget_bias=2)

print(rconv_flat.get_shape().as_list())
rnn_outputs, rnn_states = tf.nn.dynamic_rnn(lstm_cell, rconv_flat,dtype=tf.float32)

rdense = tf.layers.dense(rnn_outputs,units=1)
rprediction = tf.nn.tanh(rdense)

rnn_Error = tf.losses.mean_squared_error(labels=Ry,predictions=rprediction)
roptimizer = tf.train.AdamOptimizer(learning_rate=lr)
train_rnn = optimizer.minimize(rnn_Error)


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())


print('loading data..')
train_x, train_y = get_data('train',stacked=stack)


train_x /= 255.0


train_y = train_y.reshape(-1,1)


print('begin training for {0} steps..'.format(n_steps))
aloss = 0
for step in range(n_steps):
    b = (step*b_size)%train_x.shape[0]
    tx,ty = rnn_batchdata(train_x,train_y,max_t,b_size,step,84,84,stack)
    _,l = sess.run([train_rnn,rnn_Error],{Rx:tx,Ry:ty})
    aloss += l * 0.05
    if (step %20 == 0):
        print('step {0} | avg loss {1}'.format(step,round(aloss,3)))
        aloss = 0


print('finished')



"""
print('begin training for {0} steps..'.format(n_steps))
aloss = 0
for step in range(n_steps):
    b = (step*b_size)%train_x.shape[0]
    tx = train_x[b:b+b_size]
    ty = train_y[b:b+b_size]
    ty = ty.reshape((-1,1))
    _,l = sess.run([train_op,Error],{X:tx,Y:ty})
    aloss += l * 0.01
    if (step %100 == 0):
        print('step {0} | avg loss {1}'.format(step,round(aloss,3)))
        aloss = 0


print('finished')



"""











