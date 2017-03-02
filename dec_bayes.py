import os, sys
import numpy as np

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import sklearn.manifold
import sklearn.cluster
import matplotlib.pyplot as plt

#  model parameters
flags = tf.flags

flags.DEFINE_integer('pretrain_epochs', 500, 'Number of iterations for pretraining')
flags.DEFINE_integer('autoencoder_epochs', 100, 'Number of iterations for training autoencoder')
flags.DEFINE_integer('epochs', 100, 'Number of iterations for training clustering')
flags.DEFINE_integer('batch_size', 256, 'Size of training batches')

# parse flags
FLAGS = flags.FLAGS
FLAGS._parse_flags()

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

    def pretrain(self, sess, data, batch_size=256):
        '''
        data must have next_batch(batch_size) method
        '''
        # lets pretrain
        for l, dae in enumerate(self.autoencoders):
            for j in range(FLAGS.pretrain_epochs+1):
                batch_xs, _ = data.next_batch(batch_size)
                for i, enc in enumerate(self.autoencoders[:l]):
                    batch_xs = sess.run(enc.encode, feed_dict={enc.input_data: batch_xs})
                out = sess.run([dae.loss, dae.optimizer, dae.encode], feed_dict={dae.input_data: batch_xs})
                if j % 100 == 0:
                    print "Layer: %i, Epoch: %i, Loss: %2.4f" % (l, j, out[0])

    def train(self, sess, data, batch_size=256):
        # lets train the stacked autoencoder
        for i in range(FLAGS.autoencoder_epochs+1):
            batch_xs, _ = data.next_batch(batch_size)
            out = sess.run([self.loss, self.optimizer], feed_dict={self.input_data: batch_xs})
            if i % 100 == 0:
                print "Epoch: %i, Loss: %2.4f" % (i, out[0])

class DEC(object):
    def __init__(self, hidden_layers, n_features, n_clusters, learning_rate=0.1,
                 dropout_rate=0.0, dae_dropout_rate=0.):
        # set variables
        self.hidden_layers = hidden_layers
        self.n_features = n_features
        self.n_clusters = n_clusters

        # initialize placeholders
        self.global_step = tf.Variable(1, dtype=tf.float32)
        self.input_data = tf.placeholder(tf.float32, [None, self.n_features],
                                         name='x-input')
        self.keep_prob = tf.placeholder_with_default(1.-dropout_rate, [],
                                                    name='keep-prob')
        self.learning_rate = tf.train.exponential_decay(learning_rate, self.global_step,
                                                        20000, 0.1, staircase=True)

        # intialize autoencoder
        self.stacked_dae = StackedDAE(hidden_layers, n_features, dropout_rate=dropout_rate,
                                 learning_rate=learning_rate, dae_dropout_rate=dae_dropout_rate)
        self.encode = self.stacked_dae.encode

    def pretrain(self, sess, train_data, test_data, batch_size=256):
        print "Pretraining"
        self.stacked_dae.pretrain(sess, train_data, batch_size=batch_size)
        print "Training"
        self.stacked_dae.train(sess, train_data, batch_size=batch_size)

        # get cluster centriods
        batch_xs, _ = train_data.next_batch(batch_size*10)
        enc = sess.run(self.encode, feed_dict={self.stacked_dae.input_data: batch_xs})

        kmeans = sklearn.cluster.KMeans(n_clusters=10)
        kmeans.fit(enc)
        self.cluster_centers = kmeans.cluster_centers_  # shape=(n_clusters, hidden_layers[-1])

    def _build_graph(self, alpha=1.):
        '''
        X = (n_obs, hidden_layers[-1])
        cluster_centers = (n_clusters, hidden_layers[-1])
        q_ij = (1 + ||x_i - mu_j||^2/alpha)^(-(alpha+1)/alpha)
        q_ij /=  sum_j{(1 + ||x_i - mu_j||^2/alpha)^(-(alpha+1)/alpha)}
        use tf.scan to go through observations and compute similarities
        '''
        pass




def visualize_clusters(X):
    model = sklearn.manifold.TSNE(n_components=2, random_state=0)
    tsne_enc = model.fit_transform(X)
    plt.scatter(tsne_enc[:,0], tsne_enc[:,1])
    plt.show()

def main():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    #encoder = DAE(784, 500, dropout=0.20)
    #stacked = StackedDAE([500, 500, 2000, 10], 784, dropout_rate=0.2, dae_dropout_rate=0.20)
    dec = DEC([500, 10], 784, 10, dropout_rate=0.2, dae_dropout_rate=0.20)
    sess = tf.Session()
    with sess.as_default():
        tf.initialize_all_variables().run()
        dec.pretrain(sess, mnist.train, mnist.test)
        return
        # lets pretrain
        print "Pre-training"
        stacked.pretrain(sess, mnist.train)
        print "Training Stacked Autoencoder"
        stacked.train(sess, mnist.train)

        batch_xs, _ = mnist.train.next_batch(FLAGS.batch_size*10)
        enc = sess.run(stacked.encode, feed_dict={stacked.input_data: batch_xs})

        kmeans = sklearn.cluster.KMeans(n_clusters=10)
        kmeans.fit(enc)
        print 'Cluster centers shape:',kmeans.cluster_centers_.shape

        visualize_clusters(enc[:1000])

if __name__ == "__main__":
    main()
