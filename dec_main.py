import os, sys
import numpy as np
import pickle

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.client import timeline

import sklearn.manifold
import sklearn.cluster
import sklearn.metrics
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas as pd

#  model parameters
flags = tf.flags

flags.DEFINE_integer('dae_iters', 10001, 'Number of iterations for pretraining')
flags.DEFINE_integer('sdae_iters', 50001, 'Number of iterations for training autoencoder')
flags.DEFINE_integer('dec_epochs', 100, 'Number of iterations for training clustering')
flags.DEFINE_integer('batch_size', 256, 'Size of training batches')
flags.DEFINE_float('learning_rate', 1e-1, 'Size of training batches')
flags.DEFINE_float('momentum', 0.9, 'Momentum to use in gradient descent')
flags.DEFINE_float('dec_momentum', 0.95, 'Momentum to use in gradient descent')
flags.DEFINE_float('dae_dropout', 0.20, 'Dropout probability for denoising')
# there are 150 features
flags.DEFINE_string('layer_sizes', '10', 'Sizes of hidden layers')
flags.DEFINE_string('save_dir', '/gss_gpfs_scratch/vandal.t/data-mining-project/', 'Directory to save'\
                    'checkpoints')
flags.DEFINE_bool('use_checkpoint', True, 'try to load from checkpoint?')

# parse flags
FLAGS = flags.FLAGS
FLAGS._parse_flags()

FLAGS.layer_sizes = [int(x) for x in FLAGS.layer_sizes.split(",")]

def check_save_dir(d):
    if not os.path.exists(d):
        os.mkdir(d)

def save_trace(sess, op, feed_dict):
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    sess.run(op, feed_dict=feed_dict, options=run_options,
             run_metadata=run_metadata)
    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    with open(os.path.join(FLAGS.save_dir, 'timeline-sdae.json'), 'w') as f:
        f.write(ctf)

def _layer(inputs, W, bias, activation=tf.nn.relu, name='dense_layer'):
    y = tf.matmul(inputs, W) + bias
    if activation is not None:
        y = activation(y)
    return tf.identity(y, name=name)

class DenoisingAE(object):
    def __init__(self, inputs, units, encode_activation=tf.nn.relu,
                 decode_activation=tf.nn.relu, learning_rate=FLAGS.learning_rate):
        # get the shapes
        n_features = inputs.get_shape()[1]
        w_enc_shape = tf.stack([n_features, units])
        w_dec_shape = tf.stack([units, n_features])
        # init variable
        self.global_step = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        learning_rate = tf.train.exponential_decay(learning_rate, self.global_step, 10000, 0.1,
                                                   staircase=False)
        with tf.device("/cpu:0"):
            self.keep_prob = tf.placeholder_with_default(1.0, shape=())

        # normalize inputs
        with tf.name_scope("normalize"):
            self.inputs_norm = tf.contrib.layers.batch_norm(inputs, trainable=False)
            #self.inputs_norm = inputs

        # encode
        with tf.name_scope("encode"):
            self.W_enc = tf.Variable(tf.random_normal(shape=w_enc_shape, stddev=0.01), name='enc-w')
            self.b_enc = tf.Variable(tf.constant(0., shape=[units]), name='b-enc')
            drop_inputs = tf.nn.dropout(self.inputs_norm, keep_prob=self.keep_prob)
            self.encode = _layer(drop_inputs, self.W_enc, self.b_enc, activation=encode_activation,
                            name='encode_layer')
        # decode
        with tf.name_scope("decode"):
            self.W_dec = tf.Variable(tf.random_normal(shape=w_dec_shape, stddev=0.01), name='dec-w')
            self.b_dec = tf.Variable(tf.constant(0., shape=[n_features]), name='b-dec')
            h_dropped = tf.nn.dropout(self.encode, keep_prob=self.keep_prob)
            self.decode = _layer(h_dropped, self.W_dec, self.b_dec, activation=decode_activation,
                            name='decode_layer')

        self.loss = tf.losses.mean_squared_error(self.decode, self.inputs_norm)
        opt = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum)
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
    x, _ = data.next_batch(1)
    with tf.Graph().as_default():
        autoencoders = []
        with tf.device("/gpu:0"):
            inputs = tf.placeholder(tf.float32, shape=(None, x.shape[1]))
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
            log_path = os.path.join(FLAGS.save_dir, 'logs-pretraining')
            check_save_dir(log_path)
            writer = tf.summary.FileWriter(log_path)
            writer.add_graph(tf.get_default_graph())

            def feed_dict(enc):
                return {inputs: data.next_batch(FLAGS.batch_size)[0],
                        enc.keep_prob: 1-FLAGS.dae_dropout}

            total_steps = 0
            for k, encoder in enumerate(autoencoders):
                # figure out the mean
                for j in range(100):
                    sess.run(encoder.inputs_norm, feed_dict=feed_dict(encoder))
                for j in range(FLAGS.dae_iters):
                    feed = feed_dict(encoder)
                    out = sess.run([encoder.loss, encoder.opt], feed_dict=feed)
                    total_steps += 1
                    if j % 500 == 0:
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
    def __init__(self, init_weights, learning_rate=FLAGS.learning_rate):
        n_features = init_weights[0]['W_enc'].shape[0]
        n_layers = len(init_weights)
        with tf.name_scope("placeholders"):
            self.inputs = tf.placeholder(tf.float32, shape=(None, n_features))
        self.global_step = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        learning_rate = tf.train.exponential_decay(learning_rate, self.global_step, 10000, 0.1,
                                                   staircase=False)

        # normalize inputs
        with tf.variable_scope("normalize"):
            self.inputs_norm = tf.contrib.layers.batch_norm(self.inputs, trainable=False,
                                                           center=False, scale=False)
            tf.get_variable_scope().reuse_variables()
            self.moving_mean = tf.get_variable("BatchNorm/moving_mean")
            self.moving_variance = tf.get_variable("BatchNorm/moving_variance")
        with tf.name_scope("encoder"):
            h = self.inputs_norm
            self.W_enc, self.b_enc = [], []
            for i, p in enumerate(init_weights):
                encode_func = tf.nn.relu
                if (i+1) == n_layers:
                    encode_func = None
                with tf.name_scope("layer-%i" % i):
                    self.W_enc.append(tf.Variable(p['W_enc'], name='W', dtype=tf.float32))
                    self.b_enc.append(tf.Variable(p['b_enc'], name='b', dtype=tf.float32))
                    print "layer %i"%i, h, self.W_enc[-1]
                    h = _layer(h, self.W_enc[-1], self.b_enc[-1], activation=encode_func)
            self.encoded = h

        with tf.name_scope("decoder"):
            self.W_dec, self.b_dec = [], []
            h = self.encoded
            for i, p in enumerate(init_weights[::-1]):
                decode_func = tf.nn.relu
                if (i+1) == n_layers:
                    decode_func = None
                with tf.name_scope("layer-%i" % (n_layers-i-1)):
                    print "Layer", i, "Are all finite?", np.all(np.isfinite(p['W_dec']))
                    self.W_dec.append(tf.Variable(p['W_dec'], name='W'))
                    self.b_dec.append(tf.Variable(p['b_dec'], name='b'))
                    h = _layer(h, self.W_dec[-1], self.b_dec[-1], activation=decode_func)
            self.decoded = h

        with tf.name_scope("Loss"):
            self.loss = tf.losses.mean_squared_error(self.inputs_norm, self.decoded)
        with tf.name_scope("optimizer"):
            opt = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum)
            self.opt = opt.minimize(self.loss, global_step=self.global_step)
        with tf.device("/cpu:0"):
            tf.summary.histogram('decoded', self.decoded)
            tf.summary.histogram('inputs_norm', self.inputs_norm)
            tf.contrib.layers.summarize_tensors(tf.trainable_variables())
            tf.summary.scalar('reconstruction_loss', self.loss)
            self.summary_op = tf.summary.merge_all()

def _try_loading_checkpoint(save_dir, saver, sess):
    if FLAGS.use_checkpoint:
        try:
            checkpoint = tf.train.latest_checkpoint(save_dir)
            saver.restore(sess, checkpoint)
            print "Checkpoint", checkpoint
        except Exception as err:
            print "Warning: Could not find checkpoint", err

def stacked_ae(data):
    n_features = data.next_batch(1)[0].shape[1]

    # try to load previously trained weights
    weight_str = "-".join([str(x) for x in FLAGS.layer_sizes])
    checkpoint_dir = os.path.join(FLAGS.save_dir, 'sdae-%s' % weight_str)
    weight_file = os.path.join(checkpoint_dir, '%s-pretrain.pkl' % weight_str)
    check_save_dir(checkpoint_dir)

    if os.path.exists(weight_file) and FLAGS.use_checkpoint:
        init_weights = pickle.load(open(weight_file, 'r'))
    else:
        init_weights = pretrain_daes(data)
        pickle.dump(init_weights, open(weight_file, 'w'))

    with tf.Graph().as_default():
        with tf.device("/gpu:0"):
            encoder = StackedAutoencoder(init_weights)

        saver = tf.train.Saver()
        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_op)
            _try_loading_checkpoint(checkpoint_dir, saver, sess)
            # log directory
            log_path_base = os.path.join(FLAGS.save_dir, 'logs-sde')
            log_path = os.path.join(log_path_base, weight_str)
            check_save_dir(log_path_base)
            check_save_dir(log_path)
            writer = tf.summary.FileWriter(log_path)
            writer.add_graph(tf.get_default_graph())

            def feed_dict():
                return {encoder.inputs: data.next_batch(256)[0]}

            for j in range(100):
                sess.run(encoder.loss, feed_dict=feed_dict())

            save_trace(sess, encoder.loss, feed_dict())
           
            i = sess.run(encoder.global_step)
            for i in range(i, FLAGS.sdae_iters):
                out =sess.run([encoder.loss, encoder.opt, encoder.summary_op], feed_dict=feed_dict())
                if i % 500 == 0:
                    print "Iteration: %i, Loss: %2.5f" % (i, out[0])
                    writer.add_summary(out[2], i)
                if i % 1000 == 0:
                    save_path = saver.save(sess, os.path.join(checkpoint_dir, "model_%08i.ckpt" % i))

            encoded_data = []
            for i, (x, ltln) in enumerate(data.generate_epoch(1000)):
                encoded_data += sess.run([encoder.encoded], feed_dict={encoder.inputs: x})
            encoded_data = np.concatenate(encoded_data, axis=0)
            weights = sess.run(encoder.W_enc)
            biases = sess.run(encoder.b_enc)
            mu, var = sess.run([encoder.moving_mean, encoder.moving_variance])
    return encoded_data, weights, biases, mu, var

class DEC(object):
    def __init__(self, init_weights, init_biases, init_centriods,
                init_mean, init_var, alpha=1., learning_rate=0.1):
        self.alpha = alpha
        n_features = init_weights[0].shape[0]
        n_layers = len(init_weights)
        with tf.name_scope("placeholders"):
            self.inputs = tf.placeholder(tf.float32, shape=(None, n_features))
            with tf.device("/cpu:0"):
                self.is_training = tf.placeholder_with_default(True, shape=[])
            self.global_step = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        # normalize inputs
        with tf.name_scope("normalize"):
            self.inputs_norm = tf.contrib.layers.batch_norm(self.inputs, trainable=False,
                                                            is_training=self.is_training)
            #self.input_mean = tf.constant(init_mean)
            #self.input_var = tf.constant(init_var)
            #self.inputs_norm = tf.div(self.inputs - self.input_mean, tf.sqrt(self.input_var), name='normalize')
        with tf.name_scope("encoder"):
            h = self.inputs_norm
            self.W, self.b= [], []
            for i, (w, b) in enumerate(zip(init_weights, init_biases)):
                encode_func = tf.nn.relu
                if (i+1) == n_layers:
                    encode_func = None
                with tf.name_scope("layer-%i" % i):
                    self.W.append(tf.Variable(w, name='W'))
                    self.b.append(tf.Variable(b, name='b'))
                    h = _layer(h, self.W[-1], self.b[-1], activation=encode_func)
            self.encoded = h

        with tf.name_scope("centriods"):
            self.centriods = tf.Variable(init_centriods, name='centriods')
        with tf.name_scope("soft_dist"):
            self.q_soft = self._qi()
        with tf.name_scope("target_dist"):
            self.p_target = self._target_dist()
        with tf.name_scope("KL"):
            self.KL = tf.reduce_sum(self.p_target * tf.log(self.p_target / self.q_soft))

        self.memberships = tf.argmax(self.q_soft, axis=1)
        with tf.variable_scope("optimizer"):
            self.learning_rate = tf.train.exponential_decay(learning_rate, self.global_step, 10000, 0.1,
                                                            staircase=False)
            opt = tf.train.MomentumOptimizer(self.learning_rate, FLAGS.dec_momentum)
            #opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            self.optimize = opt.minimize(self.KL, global_step=self.global_step)
        self._summary_ops()

    def _qi(self):
        '''
        encode: tensor (n_samples, h_layer)
        cluster_centers: (n_clusters, h_layer)
        output: (n_samples, n_clusters)
        '''
        centriods = tf.expand_dims(self.centriods, 1)
        centriods = tf.tile(centriods, [1, tf.shape(self.encoded)[0], 1]) # (n_clusters, n_samples, h_layer)
        l2_squared = lambda a: tf.reduce_mean(tf.square(a), axis=2) # (n_clusters, n_samples)
        q_num = (1 + l2_squared(self.encoded-centriods) / self.alpha)**(-(self.alpha+1)/self.alpha)
        q_denom = tf.reduce_sum(q_num, axis=0) # (n_samples,) add over clusters for normalization
        return tf.transpose(q_num / q_denom)  #(n_samples, n_clusters)

    def _target_dist(self):
        # get cluster frequencies
        freq = tf.reduce_sum(self.q_soft, axis=0)  #(,n_clusters)
        q_squared = tf.square(self.q_soft)
        p_num = tf.div(q_squared, freq)  # (n_obs, n_clusters)
        p_denom = tf.reduce_sum(p_num, axis=1)  #(n_obs,)
        return tf.transpose(tf.transpose(p_num) / p_denom)

    def _summary_ops(self):
        with tf.device("/cpu:0"):
            tf.summary.histogram("inputs_norm", self.inputs_norm)
            tf.summary.scalar("KL", self.KL)
            tf.summary.scalar("learning_rate", self.learning_rate)
            [tf.summary.histogram(var.op.name, var) for var in tf.trainable_variables()]
            self.summary_op = tf.summary.merge_all()

def visualize_clusters(X, memberships=None):
    model = sklearn.manifold.TSNE(n_components=2, random_state=0)
    tsne_enc = model.fit_transform(X)
    if memberships is None:
        memberships = np.ones(tsne_enc.shape[0])
    plt.scatter(tsne_enc[:,0], tsne_enc[:,1], c=memberships, cmap=plt.cm.Accent, alpha=0.5)

def train_dec(name='dec'):
    from preprocess import ClimateData
    # try to load previously trained weights
    weight_str = "-".join([str(x) for x in FLAGS.layer_sizes])
    checkpoint_dir = os.path.join(FLAGS.save_dir, '%s-%s' % (name, weight_str))
    check_save_dir(checkpoint_dir)

    #mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    #data = mnist.train
    climatology = ClimateData('/gss_gpfs_scratch/vandal.t/data-mining-project')
    data = climatology.historical

    k_features = data.next_batch(1)[0].shape[1]
    encoded, init_weights, init_biases, mean, variance = stacked_ae(data)
    idxs = np.random.choice(range(encoded.shape[0]), 1000, replace=False)

    visualize_clusters(encoded[idxs])
    plt.savefig('tsne.pdf')

    # run kmeans on the encoded data
    idxs = np.random.choice(range(encoded.shape[0]), 50000, replace=False)
    best_dist = -1
    best_k = 9
    for k in range(best_k,best_k+1):
        #kmeans = sklearn.cluster.MiniBatchKMeans(n_clusters=k, n_init=20)
        kmeans = sklearn.cluster.KMeans(n_clusters=k, n_init=20, n_jobs=4)
        kmeans.fit(encoded)
        labels_ = kmeans.predict(encoded)
        dist = sklearn.metrics.silhouette_score(encoded[idxs], labels_[idxs])
        print "K=%i, Dist=%f" % (k, dist)
        if dist > best_dist:
            best_dist = dist
            init_centriods = kmeans.cluster_centers_
            best_k = k

    print "K=%i Clusters" % best_k
    with tf.Graph().as_default():
        with tf.device("/gpu:0"):
            dec = DEC(init_weights, init_biases, init_centriods, mean,
                      variance, learning_rate=1e-2)

        saver = tf.train.Saver()
        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_op)
            _try_loading_checkpoint(checkpoint_dir, saver, sess)
            log_path_base = os.path.join(FLAGS.save_dir, 'logs-dec')
            log_path = os.path.join(log_path_base, weight_str)
            check_save_dir(log_path_base)
            check_save_dir(log_path)
            writer = tf.summary.FileWriter(log_path)
            writer.add_graph(tf.get_default_graph())

            def feed_dict():
                return {dec.inputs: data.next_batch(256)[0]}
            for j in range(1000):
                sess.run([dec.KL], feed_dict=feed_dict())

            N = data.x.shape[0]
            for i in range(0, FLAGS.dec_epochs):
                for j, (x, ltln) in enumerate(data.generate_epoch(10000)):
                    out = sess.run([dec.KL, dec.optimize, dec.summary_op, dec.global_step],
                                   feed_dict={dec.inputs: x, dec.is_training: True})
                    if j % 10 == 0:
                        writer.add_summary(out[2], out[3])
                if i % 1 == 0:
                    print "(Training DEC) Iteration %i, KL: %2.5f" % (i, out[0])
                save_path = saver.save(sess, os.path.join(checkpoint_dir, "model_%08i.ckpt" % i))

            for d in ['historical', 'rcp45', 'rcp85']:
                locs, members = [], []
                data = getattr(climatology, d)
                silhouettes = []
                for i, (x, ltln) in enumerate(data.generate_epoch(1000)):
                    encoded, memberships = sess.run([dec.encoded, dec.memberships],
                                            feed_dict={dec.inputs: x, dec.is_training: True})
                    members.append(memberships)
                    locs.append(ltln)

                    silhouettes.append(sklearn.metrics.silhouette_score(encoded, memberships))
                print d, "Silhouettes", np.mean(silhouettes)
                members = np.concatenate(members, axis=0)[:,np.newaxis]
                locs = np.concatenate(locs, axis=0)
                clustered = np.concatenate([locs, members], axis=1)
                df = pd.DataFrame(clustered, columns=['lat','lon','membership'])
                df = pd.pivot_table(df, values='membership', columns='lon', index='lat')
                df.to_csv("output/dec%s_%s.csv" % (weight_str, d))
                plt.imshow(df.values[::-1], cmap=plt.cm.Accent)
                plt.savefig("figures/mapped_clusters_%s_%s_%i.pdf" % (weight_str, d, best_k))

if __name__ == "__main__":
    check_save_dir(FLAGS.save_dir)
    train_dec()
