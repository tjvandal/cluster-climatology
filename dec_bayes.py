import os, sys
import numpy as np

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import sklearn.manifold
import sklearn.cluster
import matplotlib.pyplot as plt

#  model parameters
flags = tf.flags

flags.DEFINE_integer('dae_iters', 10000, 'Number of iterations for pretraining')
flags.DEFINE_integer('sdae_iters', 1000, 'Number of iterations for training autoencoder')
flags.DEFINE_integer('dec_iters', 1000, 'Number of iterations for training clustering')
flags.DEFINE_integer('batch_size', 256, 'Size of training batches')
flags.DEFINE_float('learning_rate', 1e-2, 'Size of training batches')
flags.DEFINE_string('save_dir', '/home/vandal.t/repos/DEC-Bayes/models/', 'Directory to save'\
                    'checkpoints')

# parse flags
FLAGS = flags.FLAGS
FLAGS._parse_flags()

class DAE(object):
    def __init__(self, n_features, n_hidden, dropout=0.0, learning_rate=0.10,
                encode_nonlinear=True, decode_nonlinear=True, iters=100):
        # initialize parameters
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.encode_nonlinear = encode_nonlinear
        self.decode_nonlinear = decode_nonlinear
        self.iters=iters
        self.global_step = tf.Variable(0.0, dtype=tf.float32, trainable=False)

        # set placeholders
        self.input_data = tf.placeholder(tf.float32, [None, n_features],
                                         name='x-input')
        self.keep_prob = tf.placeholder_with_default(1.-dropout, [],
                                                    name='keep-prob')
        self.learning_rate = tf.train.exponential_decay(learning_rate, self.global_step,
                                                        20000, 0.1, staircase=True)

        with tf.device("/gpu:0"):
            self._build_graph()
            self.loss = tf.div(tf.reduce_mean(tf.square(self.input_data - self.reconstruction)),
                               2, name='loss')
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss,
                                                           global_step=self.global_step)

    def _build_graph(self):
        # set up variables
        self.W_enc = tf.Variable(tf.random_normal(shape=[self.n_features, self.n_hidden],
                                                 stddev=0.1), name='enc-w')
        self.W_dec = tf.Variable(tf.random_normal(shape=[self.n_hidden, self.n_features],
                                                 stddev=0.1), name='dec-w')
        self.b_enc = tf.Variable(tf.constant(0., shape=[self.n_hidden]), name='b-enc')
        self.b_dec = tf.Variable(tf.constant(0., shape=[self.n_features]), name='b-dec')

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
                 dropout_rate=0.0, dae_dropout_rate=0., iters=100,
                pretrain_iters=100):
        # initialize variables
        self.hidden_layers = hidden_layers
        self.n_features = n_features
        self.iters = iters
        self.pretrain_iters = pretrain_iters

        # initialize placeholders
        self.global_step = tf.Variable(0.0, dtype=tf.float32, name='dae_global_step', trainable=False)
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
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,
                                                           global_step=self.global_step)
        with tf.device("/cpu:0"):
            tf.summary.scalar('reconstruction_error', self.loss)
            [tf.summary.histogram(var.op.name, var) for var in tf.trainable_variables()]
            self.summary_op = tf.summary.merge_all()

    def _build_graph(self):
        self._build_daes()
        with tf.name_scope("encoder"):
            self._build_encoder()
        with tf.name_scope("decoder"):
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
            with tf.device("/cpu:0"), tf.variable_scope("sdae-%i" % j):
                self.autoencoders.append(DAE(input_dims[j], h, 1.-self.keep_prob,
                         encode_nonlinear=encode_nonlinear, decode_nonlinear=decode_nonlinear,
                                             iters=self.pretrain_iters))

    def _build_encoder(self):
        self.encode = self.input_data
        for i, autoencoder in enumerate(self.autoencoders):
            with tf.variable_scope('encode-%i' % i):
                x_dropped = tf.nn.dropout(self.encode, self.keep_prob)
                self.encode = tf.nn.relu(tf.add(tf.matmul(x_dropped, autoencoder.W_enc), autoencoder.b_enc), 'activation')

    def _build_decoder(self):
        self.reconstruction = self.encode
        for i, autoencoder in enumerate(self.autoencoders[::-1]):
            with tf.variable_scope('decode-%i' % i):
                x_dropped = tf.nn.dropout(self.reconstruction, self.keep_prob)
                self.reconstruction = tf.nn.relu(tf.add(tf.matmul(x_dropped, autoencoder.W_dec),
                                                autoencoder.b_dec), 'activation')

    def pretrain(self, sess, data, batch_size=256):
        #data must have next_batch(batch_size) method
        # lets pretrain
        for l, dae in enumerate(self.autoencoders):
            step_start = sess.run(dae.global_step)
            print "Start pretraining layer %i at Step %i" % (l, step_start)
            for j in range(step_start, self.pretrain_iters+1):
                batch_xs, _ = data.next_batch(batch_size)
                for i, enc in enumerate(self.autoencoders[:l]):
                    batch_xs = sess.run(enc.encode, feed_dict={enc.input_data: batch_xs})
                out = sess.run([dae.loss, dae.optimizer, dae.encode], feed_dict={dae.input_data: batch_xs})
                if j % 10 == 0:
                    print "Layer: %i, Epoch: %i, Loss: %2.4f" % (l, j, out[0])

    def train(self, sess, data, batch_size=256):
        # lets train the stacked autoencoder
        step_start = sess.run(self.global_step)
        print "Trainiable Variables:", [var.op.name for var in tf.trainable_variables()]
        print "Starting SDAE at step:", step_start
        writer = tf.summary.FileWriter("./logs-sde/")
        for i in range(step_start, self.iters+1):
            batch_xs, _ = data.next_batch(batch_size)
            out = sess.run([self.loss, self.optimizer, self.summary_op], feed_dict={self.input_data: batch_xs})
            if i % 10 == 0:
                print "Epoch: %i, Loss: %2.4f" % (i, out[0])
                writer.add_summary(out[2], global_step=i)

class DEC(object):
    def __init__(self, hidden_layers, n_features, n_clusters, learning_rate=0.1,
                 dropout_rate=0.0, dae_dropout_rate=0., alpha=1., iters=100,
                sdae_iters=100, dae_iters=100):
        # set variables
        self.hidden_layers = hidden_layers
        self.n_features = n_features
        self.alpha = alpha
        self.n_clusters = n_clusters
        self.iters = iters

        self.global_step = tf.Variable(0.0, dtype=tf.float32, name='global_step', trainable=False)
        with tf.name_scope("placeholders"):
            self.keep_prob = tf.placeholder_with_default(1.-dropout_rate, [],
                                                        name='keep-prob')
            self.learning_rate = tf.train.exponential_decay(learning_rate, self.global_step,
                                                        20000, 0.1, staircase=True)

        # intialize autoencoder
        with tf.name_scope("StackedDAE"):
            self.stacked_dae = StackedDAE(hidden_layers, n_features, dropout_rate=dropout_rate,
                                 learning_rate=learning_rate, dae_dropout_rate=dae_dropout_rate,
                                         iters=sdae_iters, pretrain_iters=dae_iters)

        with tf.device("/gpu:0"):
            self._build_graph()

    def pretrain(self, sess, train_data, test_data, batch_size=256):
        print "Pretraining"
        self.stacked_dae.pretrain(sess, train_data, batch_size=batch_size)

        print "Training"
        self.stacked_dae.train(sess, train_data, batch_size=batch_size)

        # get cluster centriods
        batch_xs, _ = train_data.next_batch(batch_size*10)
        enc = sess.run(self.stacked_dae.encode, feed_dict={self.stacked_dae.input_data: batch_xs})

        # if we do a loop we can figure out the number of clusters here
        kmeans = sklearn.cluster.KMeans(n_clusters=10)
        kmeans.fit(enc)
        tf.assign(self.cluster_centers, kmeans.cluster_centers_)

    def _qi(self, _, x):
        '''
        x: tensor (n_features,)
        cluster_centers: (n_clusters, n_features)
        '''
        centriods = self.cluster_centers
        l2_squared = lambda a: tf.reduce_mean(tf.square(a), axis=1)
        q_num = (1 + l2_squared(x-centriods) / self.alpha)**(-(self.alpha+1)/self.alpha)
        q_denom = tf.reduce_sum(q_num)
        return tf.reshape(q_num / q_denom, (1,self.n_clusters))

    def _qi2(self):
        '''
        encode: tensor (n_samples, h_layer)
        cluster_centers: (n_clusters, h_layer)
        output: (n_samples, n_clusters)
        '''
        centriods = tf.expand_dims(self.cluster_centers, 1)
        x = self.encode
        centriods = tf.tile(centriods, [1, tf.shape(x)[0], 1]) # (n_clusters, n_samples, h_layer)
        l2_squared = lambda a: tf.reduce_mean(tf.square(a), axis=2) # (n_clusters, n_samples)
        q_num = (1 + l2_squared(x-centriods) / self.alpha)**(-(self.alpha+1)/self.alpha)
        q_denom = tf.reduce_sum(q_num, axis=0) # (n_samples,) add over clusters for normalization
        return tf.transpose(q_num / q_denom)  #(n_samples, n_clusters)

    def _target_dist(self):
        # get cluster frequencies
        with tf.name_scope("target_dist"):
            freq = tf.reduce_sum(self.q_soft, axis=0)  #(,n_clusters)
            q_squared = tf.square(self.q_soft)
            p_num = tf.div(q_squared, freq)  # (n_obs, n_clusters)
            p_denom = tf.reduce_sum(p_num, axis=1)  #(n_obs,)
            return tf.transpose(tf.transpose(p_num) / p_denom)

    def _build_graph(self):
        #### 
        # X = (n_obs, hidden_layers[-1])
        # cluster_centers = (n_clusters, hidden_layers[-1])
        # q_ij = (1 + ||x_i - mu_j||^2/alpha)^(-(alpha+1)/alpha)
        # q_ij /=  sum_j{(1 + ||x_i - mu_j||^2/alpha)^(-(alpha+1)/alpha)}
        # use tf.scan to go through observations and compute similarities
        # p_ij = q_ij^2 / f_j
        # p_ij /= sum_j{q_ij^2 / f_j}
        ####
        self.cluster_centers = tf.Variable(tf.zeros(shape=(self.n_clusters, self.hidden_layers[-1])),
                                           name='centriods', trainable=True)

        self.encode = self.stacked_dae.encode
        with tf.name_scope("soft_dist"):
            self.q_soft = self._qi2()

        self.p_target = self._target_dist()
        with tf.name_scope("KL"):
            self.KL = tf.reduce_sum(self.p_target * tf.log(self.p_target / self.q_soft))

        train_vars = [var.op.name for var in tf.trainable_variables()]
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.optimize = opt.minimize(self.KL, global_step=self.global_step)
        self._summary_ops()

    def _summary_ops(self):
        with tf.device("/cpu:0"):
            tf.summary.scalar('input_data', tf.reduce_mean(self.stacked_dae.input_data))
            tf.summary.scalar("KL", self.KL)
            #tf.contrib.layers.summarize_tensors(tf.trainable_variables())
            [tf.summary.histogram(var.op.name, var) for var in tf.trainable_variables()]
            self.summary_op = tf.summary.merge_all()

    def train(self, sess, train_data, batch_size=256, save_dir=None):
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter("./logs/")
        writer.add_graph(tf.get_default_graph())

        batch_xs, _ = train_data.next_batch(batch_size)
        feed_dict = {self.stacked_dae.input_data: batch_xs}

        out = sess.run(self.summary_op, feed_dict=feed_dict)
        step_start = int(sess.run(self.global_step))
        print "Starting DEC at step %i" % step_start
        for k in range(step_start, self.iters+1):
            res = sess.run([self.KL, self.optimize, self.global_step, self.q_soft, self.summary_op], feed_dict=feed_dict)
            if k % 10 == 0:
                writer.add_summary(res[-1], global_step=k)
                print "(Training DEC) Iteration: %i, KL: %2.8f" % (k, res[0])

def visualize_clusters(X):
    model = sklearn.manifold.TSNE(n_components=2, random_state=0)
    tsne_enc = model.fit_transform(X)
    plt.scatter(tsne_enc[:,0], tsne_enc[:,1])
    plt.show()

def main():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    dec = DEC([500,], 784, 10, dropout_rate=0.0, dae_dropout_rate=0.20,
              learning_rate=FLAGS.learning_rate, iters=FLAGS.dec_iters,
             dae_iters=FLAGS.dae_iters, sdae_iters=FLAGS.sdae_iters)
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init_op)
        # lets see if there is a checkpoint
        try:
            checkpoint = tf.train.latest_checkpoint(FLAGS.save_dir)
            #saver.restore(sess, checkpoint)
            print "Checkpoint found:", checkpoint
        except:
            print "Warning could not find checkpoint"

        print "Pre-training"
        dec.pretrain(sess, mnist.train, mnist.test)
        print "Training DEC"
        dec.train(sess, mnist.train)

        # lets save what we trained
        save_path = saver.save(sess, os.path.join(FLAGS.save_dir, "dec-model.ckpt"))

        return
        visualize_clusters(enc[:1000])

if __name__ == "__main__":
    main()
