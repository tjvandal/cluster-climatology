import os, sys
import numpy as np
import pickle

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import sklearn.manifold
import sklearn.cluster
import matplotlib.pyplot as plt

#  model parameters
flags = tf.flags

flags.DEFINE_integer('dae_iters', 5000, 'Number of iterations for pretraining')
flags.DEFINE_integer('sdae_iters', 5000, 'Number of iterations for training autoencoder')
flags.DEFINE_integer('dec_iters', 1000, 'Number of iterations for training clustering')
flags.DEFINE_integer('batch_size', 256, 'Size of training batches')
flags.DEFINE_float('learning_rate', 1e-1, 'Size of training batches')
flags.DEFINE_string('layer_sizes', '500,500,2000,10', 'Sizes of hidden layers')
flags.DEFINE_string('save_dir', '/home/vandal.t/repos/DEC-Bayes/models/', 'Directory to save'\
                    'checkpoints')

# parse flags
FLAGS = flags.FLAGS
FLAGS._parse_flags()

FLAGS.layer_sizes = [int(x) for x in FLAGS.layer_sizes.split(",")]

def _layer(inputs, W, bias, activation=tf.nn.relu, name='dense_layer'):
    y = tf.matmul(inputs, W) + bias
    if activation is not None:
        y = activation(y)
    return tf.identity(y, name=name)

class DenoisingAE(object):
    def __init__(self, inputs, units, encode_activation=tf.nn.relu,
                 decode_activation=tf.nn.relu):
        # get the shapes
        n_features = inputs.get_shape()[1]
        w_enc_shape = tf.stack([n_features, units])
        w_dec_shape = tf.stack([units, n_features])

        # init variable
        self.global_step = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        with tf.device("/cpu:0"):
            self.keep_prob = tf.placeholder_with_default(1.0, shape=())

        # encode
        with tf.name_scope("encode"):
            self.W_enc = tf.Variable(tf.random_normal(shape=w_enc_shape, stddev=0.01), name='enc-w')
            self.b_enc = tf.Variable(tf.constant(0., shape=[units]), name='b-enc')
            drop_inputs = tf.nn.dropout(inputs, keep_prob=self.keep_prob)
            self.encode = _layer(drop_inputs, self.W_enc, self.b_enc, activation=encode_activation,
                            name='encode_layer')
        # decode
        with tf.name_scope("decode"):
            self.W_dec = tf.Variable(tf.random_normal(shape=w_dec_shape, stddev=0.01), name='dec-w')
            self.b_dec = tf.Variable(tf.constant(0., shape=[n_features]), name='b-dec')
            h_dropped = tf.nn.dropout(self.encode, keep_prob=self.keep_prob)
            self.decode = _layer(h_dropped, self.W_dec, self.b_dec, activation=decode_activation,
                            name='decode_layer')

        self.loss = tf.losses.mean_squared_error(self.decode, inputs)
        #opt = tf.train.AdamOptimizer(0.1)
        opt = tf.train.MomentumOptimizer(0.1, 0.9)
        self.opt = opt.minimize(self.loss, var_list=[self.W_enc, self.W_dec, self.b_enc,
                                     self.b_dec], global_step=self.global_step)
        with tf.device("/cpu:0"):
            summaries = [tf.summary.scalar('loss', self.loss),
                        tf.summary.histogram('W-enc', self.W_enc),
                        tf.summary.histogram('W-dec', self.W_dec),
                        tf.summary.histogram('b-enc', self.b_enc),
                        tf.summary.histogram('b-dec', self.b_dec),
                        ]

            self.summary_op = tf.summary.merge(summaries)

def pretrain_daes(data):
    layer_sizes = FLAGS.layer_sizes
    with tf.Graph().as_default():
        inputs = tf.placeholder(tf.float32, shape=(None, 784))
        autoencoders = []
        with tf.device("/gpu:0"):
            for i, units in enumerate(layer_sizes):
                encode_func, decode_func = tf.nn.relu, tf.nn.relu
                if i == 0:
                    h = inputs
                    decode_func = None
                else:
                    h = autoencoders[i-1].encode

                if i == (len(layer_sizes)-1):
                    encode_func = None

                with tf.name_scope("dae-%i" % i):
                    print "layer", i, h, units
                    autoencoders.append(DenoisingAE(h, units,
                                           encode_activation=encode_func,
                                           decode_activation=decode_func))

        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_op)
            writer = tf.summary.FileWriter("./logs-pretraining/")
            writer.add_graph(tf.get_default_graph())

            def feed_dict(enc):
                return {inputs: data.next_batch(FLAGS.batch_size)[0],
                        enc.keep_prob: 0.8}

            total_steps = 0
            for k, encoder in enumerate(autoencoders):
                for j in range(FLAGS.dae_iters):
                    feed = feed_dict(encoder)
                    out = sess.run([encoder.loss, encoder.opt], feed_dict=feed)
                    total_steps += 1
                    if j % 10 == 0:
                        print "(Pre-training Layer %i) Iteration: %i, Loss: %2.9f" % (k, j, out[0])
                        summaries = [enc.summary_op for enc in autoencoders[:k+1]]
                        summaries = sess.run(summaries, feed_dict=feed)
                        [writer.add_summary(s, total_steps) for s in summaries]
            params = []
            for enc in autoencoders:
                out = sess.run([enc.W_enc, enc.b_enc, enc.W_dec, enc.b_dec])
                params.append(dict(W_enc=out[0], b_enc=out[1], W_dec=out[2], b_dec=out[3]))
        return params

class StackedAutoencoder(object):
    def __init__(self, init_weights):
        n_features = init_weights[0]['W_enc'].shape[0]
        n_layers = len(init_weights)
        with tf.name_scope("placeholders"):
            self.inputs = tf.placeholder(tf.float32, shape=(None, n_features))
            self.global_step = tf.Variable(0.0, dtype=tf.float32, trainable=False)

        with tf.name_scope("encoder"):
            h = self.inputs
            self.W_enc, self.b_enc = [], []
            for i, p in enumerate(init_weights):
                encode_func = tf.nn.relu
                if (i+1) == n_layers:
                    encode_func = None
                with tf.name_scope("layer-%i" % i):
                    self.W_enc.append(tf.Variable(p['W_enc'], name='W'))
                    self.b_enc.append(tf.Variable(p['b_enc'], name='b'))
                    h = _layer(h, self.W_enc[-1], self.b_enc[-1], activation=encode_func)
            self.encoded = h

        with tf.name_scope("decoder"):
            self.W_dec, self.b_dec = [], []
            h = self.encoded
            for i, p in enumerate(init_weights[::-1]):
                decode_func = tf.nn.relu
                if (i+1) == n_layers:
                    decode_func = None
                with tf.name_scope("layer-%i" % (n_layers-i)):
                    self.W_dec.append(tf.Variable(p['W_dec'], name='W'))
                    self.b_dec.append(tf.Variable(p['b_dec'], name='b'))
                    h = _layer(h, self.W_dec[-1], self.b_dec[-1], activation=decode_func)
            self.decoded = h

        with tf.name_scope("Loss"):
            self.loss = tf.losses.mean_squared_error(self.inputs, self.decoded)
        with tf.name_scope("optimizer"):
            #opt = tf.train.AdamOptimizer(1e-2)
            opt = tf.train.MomentumOptimizer(0.1, 0.9)
            self.opt = opt.minimize(self.loss, global_step=self.global_step)
        with tf.device("/cpu:0"):
            tf.contrib.layers.summarize_tensors(tf.trainable_variables())
            tf.summary.scalar('reconstruction_loss', self.loss)
            input_image = tf.reshape(self.inputs, [-1, 28, 28, 1])
            recon_image = tf.reshape(self.decoded, [-1, 28, 28, 1])
            tf.summary.image('input-mnist', input_image)
            tf.summary.image('reconstruction', recon_image)
            self.summary_op = tf.summary.merge_all()

def stacked_ae():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    data = mnist.train
    n_features = data.next_batch(1)[0].shape[1]

    # try to load previously trained weights
    weight_file = "-".join([str(x) for x in FLAGS.layer_sizes]) + "-pretrain.pkl"
    weight_file = os.path.join("models", weight_file)
    if os.path.exists(weight_file):
        init_weights = pickle.load(open(weight_file, 'r'))
    else:
        init_weights = pretrain_daes(data)
        pickle.dump(init_weights, open(weight_file, 'w'))

    with tf.Graph().as_default():
        with tf.device("/gpu:0"):
            encoder = StackedAutoencoder(init_weights)

        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_op)
            writer = tf.summary.FileWriter("./logs-sde/")
            writer.add_graph(tf.get_default_graph())
            def feed_dict():
                return {encoder.inputs: data.next_batch(256)[0]}

            for i in range(FLAGS.sdae_iters):
                out =sess.run([encoder.loss, encoder.opt, encoder.summary_op], feed_dict=feed_dict())
                if i % 10 == 0:
                    print "Iteration: %i, Loss: %2.5f" % (i, out[0])
                    writer.add_summary(out[2], i)

if __name__ == "__main__":
    stacked_ae()
