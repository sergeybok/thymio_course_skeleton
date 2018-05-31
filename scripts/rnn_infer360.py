import tensorflow as tf
import numpy as np

from Data import get_data
import os, cv2, re

n_steps = 1000
lr = 0.0004
b_size = 40

stack = 2

max_t = 6
rnn_hid = 8

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
    conv1 = tf.layers.conv2d(small_x,filters=16,kernel_size=[7,7],padding='valid',activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(conv1,pool_size=[4,4],strides=[4,4])
    conv2 = tf.layers.conv2d(pool1, filters=10,kernel_size=[2,2],padding='valid',activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(conv2,pool_size=[3,3],strides=[3,3])
    #conv3 = tf.layers.conv2d(pool2,filters=8, kernel_size=[3,3],padding='valid')
    #pool3 = tf.layers.max_pooling2d(conv3,pool_size=[2,2],strides=[2,2])
    return pool2
    #return conv3

class RNNDriver(object):

    def __init__(self):
        self.build_graph()
        self.init_session()
        self.last_hid_state = self.sess.run(self.zero_state,{})

    def build_graph(self):

        X = tf.placeholder(tf.float32,[None,84,84,stack])

        conv_out = CNN(X)

        conv_flat = tf.contrib.layers.flatten(conv_out)
        dense = tf.layers.dense(conv_flat, units=1)
        prediction = tf.nn.tanh(dense)

        rnn_batch = 1
        max_t = 1
      
        self.Rx = tf.placeholder(tf.float32,[max_t,1,84,84,stack])
        self.Ry = tf.placeholder(tf.float32,[max_t,1,1])

        rnn_cnn_input = tf.reshape(self.Rx,[max_t*rnn_batch,84,84,stack])

        rconv_out = CNN(rnn_cnn_input)

        rconv_shape = rconv_out.get_shape().as_list()
        rnn_conv_out = tf.reshape(rconv_out,[max_t,rnn_batch,rconv_shape[1],rconv_shape[2],rconv_shape[3]])
        rconv_flat = tf.reshape(rconv_out,[max_t,rnn_batch,rconv_shape[1]*rconv_shape[2]*rconv_shape[3]])

        print(rconv_flat.get_shape().as_list())

        lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=rnn_hid,forget_bias=2,state_is_tuple=False)

        self.zero_state = lstm_cell.zero_state(rnn_batch,dtype=tf.float32)
        
        self.hid_state = tf.placeholder_with_default(self.zero_state,shape=[1,2*rnn_hid])
        
        
        rnn_outputs, self.rnn_states = tf.nn.dynamic_rnn(cell=lstm_cell,
            inputs=rconv_flat,
            initial_state=self.hid_state,
            dtype=tf.float32,
            time_major=True)

        rdense = tf.layers.dense(rnn_outputs,units=1)
        self.prediction = tf.nn.tanh(rdense)

        #rnn_Error = tf.losses.mean_squared_error(labels=Ry,predictions=rprediction)

    def init_session(self,model_file='cnnout-360_test-0,21/model'):
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(self.sess,model_file)
    
    def infer(self,imgs):
        pred, self.last_hid_state = self.sess.run([self.prediction,self.rnn_states],{self.Rx:imgs,self.hid_state:self.last_hid_state})
        #self.last_hid_state = cur_state.reshape((rnn_hid,rnn_hid))
        return pred

