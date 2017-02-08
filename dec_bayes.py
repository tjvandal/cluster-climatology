import os, sys
import numpy as np

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class DAE(object):
    def __init__(self, n_features, n_hidden, dropout=0.0, learning_rate=0.10, 
                encode_nonlinear=True, decode_nonlinear=True):
        # initialize parameters
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.encode_nonlinear = encode_nonlinear
        self.decode_nonlinear = decode_nonlinear
        self.global_step = tf.Variable(1, dtype=tf.float32, trainable=False)

        # set placeholders
        self.input_data = tf.placeholder(tf.float32, [None, n_features],
                                         name='x-input')
        self.keep_prob = tf.placeholder_with_default(1.-dropout, [],
                                                    name='keep-prob')
        self.learning_rate = tf.train.exponential_decay(learning_rate, self.global_step,
                                                        20000, 0.1, staircase=True)

        with tf.device("/gpu:0"):
            self._build_graph()

            # loss
            self.loss = tf.div(tf.reduce_mean(tf.square(self.input_data - self.reconstruction)),
                               2, name='loss')
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss,
                                                                   global_step=self.global_step)

    def _build_graph(self):
        # set up variables
        self.W_enc = tf.Variable(tf.random_normal(shape=[self.n_features, self.n_hidden],
                                                 stddev=0.01), name='enc-w')
        self.W_dec = tf.Variable(tf.random_normal(shape=[self.n_hidden, self.n_features],
                                                 stddev=0.01), name='dec-w')
        self.b_enc = tf.Variable(tf.constant(0., shape=[self.n_hidden]))
        self.b_dec = tf.Variable(tf.constant(0., shape=[self.n_features]))

        # build encoder and decoder
        self._build_encoder()
        self._build_decoder()

    def _build_encoder(self):
        with tf.name_scope("encoder"):
            x_dropped = tf.nn.dropout(self.input_data, self.keep_prob)
            if self.encode_nonlinear:
                self.encode = tf.nn.relu(tf.add(tf.matmul(x_dropped, self.W_enc), self.b_enc), 'activation')
            else:
                self.encode = tf.add(tf.matmul(x_dropped, self.W_enc), self.b_enc, name='activation')

    def _build_decoder(self):
        with tf.name_scope("decoder"):
            enc_dropped = tf.nn.dropout(self.encode, self.keep_prob)
            if self.decode_nonlinear:
                self.reconstruction = tf.nn.relu(tf.add(tf.matmul(enc_dropped, self.W_dec), self.b_dec),
                                             name='reconstruction')
            else:
                self.reconstruction = tf.add(tf.matmul(enc_dropped, self.W_dec), self.b_dec,
                                             name='reconstruction')

class StackedDAE(object):
    def __init__(self, hidden_layers, n_features, learning_rate=0.1,
                 dropout_rate=0.0, dae_dropout_rate=0.):
        # initialize variables
        self.hidden_layers = hidden_layers
        self.n_features = n_features

        # initialize placeholders
        self.global_step = tf.Variable(1, dtype=tf.float32)
        self.input_data = tf.placeholder(tf.float32, [None, self.n_features],
                                         name='x-input')
        self.keep_prob = tf.placeholder_with_default(1.-dropout_rate, [],
                                                    name='keep-prob')
        self.learning_rate = tf.train.exponential_decay(learning_rate, self.global_step,
                                                        20000, 0.1, staircase=True)
        # build graph
        with tf.device("/gpu:0"):
            self._build_graph()

            self.loss = tf.div(tf.reduce_mean(tf.square(self.input_data - self.reconstruction)),
                               2, name='loss')
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss,
                                                                   global_step=self.global_step)

    def _build_graph(self):
        self._build_daes()
        self._build_encoder()
        self._build_decoder()

    def _build_daes(self):
        input_dims = [self.n_features] + self.hidden_layers
        self.autoencoders = []
        for j, h in enumerate(self.hidden_layers):
            encode_nonlinear, decode_nonlinear = True, True
            if j == 0:
                decode_nonlinear = False
            if j == (len(self.hidden_layers)-1):
                encode_nonlinear = False
            with tf.device("/cpu:0"):
                self.autoencoders.append(DAE(input_dims[j], h, 1.-self.keep_prob,
                         encode_nonlinear=encode_nonlinear, decode_nonlinear=decode_nonlinear))

    def _build_encoder(self):
        self.encode = self.input_data
        for i, autoencoder in enumerate(self.autoencoders):
            with tf.name_scope('encode-%i' % i):
                x_dropped = tf.nn.dropout(self.encode, self.keep_prob)
                self.encode = tf.nn.relu(tf.add(tf.matmul(x_dropped, autoencoder.W_enc), autoencoder.b_enc), 'activation')

    def _build_decoder(self):
        self.reconstruction = self.encode
        for i, autoencoder in enumerate(self.autoencoders[::-1]):
            with tf.name_scope('decode-%i' % i):
                x_dropped = tf.nn.dropout(self.reconstruction, self.keep_prob)
                self.reconstruction = tf.nn.relu(tf.add(tf.matmul(x_dropped, autoencoder.W_dec),
                                                autoencoder.b_dec), 'activation')

def main():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    #encoder = DAE(784, 500, dropout=0.20)
    stacked = StackedDAE([500, 500], 784, dropout_rate=0.2)
    sess = tf.Session()
    with sess.as_default():
        tf.initialize_all_variables().run()
        # lets pretrain
        for l, dae in enumerate(stacked.autoencoders):
            for j in range(5000):
                batch_xs, _ = mnist.train.next_batch(256)
                for i, enc in enumerate(stacked.autoencoders[:l]):
                    batch_xs = sess.run(enc.encode, feed_dict={enc.input_data: batch_xs})
                out = sess.run([dae.loss, dae.optimizer, dae.encode], feed_dict={dae.input_data: batch_xs})
                if j % 100 == 0:
                    print "Layer: %i, Epoch: %i, Loss: %2.4f" % (l, j, out[0])

if __name__ == "__main__":
    main()
