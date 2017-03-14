import os, sys
import numpy as np
import pickle

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import sklearn.manifold
import sklearn.cluster
import sklearn.metrics
import matplotlib.pyplot as plt

#  model parameters
flags = tf.flags

flags.DEFINE_integer('dae_iters', 50000, 'Number of iterations for pretraining')
flags.DEFINE_integer('sdae_iters', 101, 'Number of iterations for training autoencoder')
flags.DEFINE_integer('dec_iters', 100, 'Number of iterations for training clustering')
flags.DEFINE_integer('batch_size', 256, 'Size of training batches')
flags.DEFINE_float('learning_rate', 1e-1, 'Size of training batches')
flags.DEFINE_string('layer_sizes', '500,500,2000,10', 'Sizes of hidden layers')
flags.DEFINE_string('save_dir', '/home/vandal.t/repos/DEC-Bayes/models/', 'Directory to save'\
                    'checkpoints')

# parse flags
FLAGS = flags.FLAGS
FLAGS._parse_flags()

FLAGS.layer_sizes = [int(x) for x in FLAGS.layer_sizes.split(",")]

def check_save_dir(d):
    if not os.path.exists(d):
        os.mkdir(d)

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
		learning_rate = tf.train.exponential_decay(learning_rate, self.global_step, 20000, 0.1, staircase=True)
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
		opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
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
    def __init__(self, init_weights, learning_rate=FLAGS.learning_rate):
		n_features = init_weights[0]['W_enc'].shape[0]
		n_layers = len(init_weights)
		with tf.name_scope("placeholders"):
			self.inputs = tf.placeholder(tf.float32, shape=(None, n_features))
		self.global_step = tf.Variable(0.0, dtype=tf.float32, trainable=False)
		learning_rate = tf.train.exponential_decay(learning_rate, self.global_step, 20000, 0.1, staircase=True)

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
			opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
			self.opt = opt.minimize(self.loss, global_step=self.global_step)
		with tf.device("/cpu:0"):
			tf.contrib.layers.summarize_tensors(tf.trainable_variables())
			tf.summary.scalar('reconstruction_loss', self.loss)
			input_image = tf.reshape(self.inputs, [-1, 28, 28, 1])
			recon_image = tf.reshape(self.decoded, [-1, 28, 28, 1])
			tf.summary.image('input-mnist', input_image)
			tf.summary.image('reconstruction', recon_image)
			self.summary_op = tf.summary.merge_all()

def _try_loading_checkpoint(save_dir, saver, sess):
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
	weight_file = os.path.join("models", '%s-pretrain.pkl' % weight_str)
	checkpoint_dir = os.path.join(FLAGS.save_dir, 'sdae-%s' % weight_str)
	check_save_dir(checkpoint_dir)

	if os.path.exists(weight_file):
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
			writer = tf.summary.FileWriter("./logs-sde/")
			writer.add_graph(tf.get_default_graph())

			def feed_dict():
				return {encoder.inputs: data.next_batch(256)[0]}

			i = sess.run(encoder.global_step)
			for i in range(i, FLAGS.sdae_iters):
				out =sess.run([encoder.loss, encoder.opt, encoder.summary_op], feed_dict=feed_dict())
				if i % 10 == 0:
					print "Iteration: %i, Loss: %2.5f" % (i, out[0])
					writer.add_summary(out[2], i)
			save_path = saver.save(sess, os.path.join(checkpoint_dir, "model_%08i.ckpt" % i))

			encode_data = sess.run(encoder.encoded, feed_dict={encoder.inputs: data.next_batch(FLAGS.batch_size*10)[0]})
			weights = sess.run(encoder.W_enc)
			biases = sess.run(encoder.b_enc)
	return encode_data, weights, biases

class DEC(object):
    def __init__(self, init_weights, init_biases, init_centriods, alpha=1.,
                learning_rate=0.1):
		self.alpha = alpha
		n_features = init_weights[0].shape[0]
		n_layers = len(init_weights)
		with tf.name_scope("placeholders"):
			self.inputs = tf.placeholder(tf.float32, shape=(None, n_features))
			self.global_step = tf.Variable(0.0, dtype=tf.float32, trainable=False)

		with tf.name_scope("encoder"):
			h = self.inputs
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

		self.learning_rate = tf.train.exponential_decay(learning_rate, self.global_step, 20000, 0.1, staircase=True)
		opt = tf.train.MomentumOptimizer(self.learning_rate, 0.9)
		self.optimize = opt.minimize(self.KL, global_step=self.global_step)
		self._summary_ops()

    def _qi(self):
        '''
        encode: tensor (n_samples, h_layer)
        cluster_centers: (n_clusters, h_layer)
        output: (n_samples, n_clusters)
        '''
        centriods = tf.expand_dims(self.centriods, 1)
        x = self.encoded
        centriods = tf.tile(centriods, [1, tf.shape(x)[0], 1]) # (n_clusters, n_samples, h_layer)
        l2_squared = lambda a: tf.reduce_mean(tf.square(a), axis=2) # (n_clusters, n_samples)
        q_num = (1 + l2_squared(x-centriods) / self.alpha)**(-(self.alpha+1)/self.alpha)
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
            tf.summary.scalar("KL", self.KL)
            [tf.summary.histogram(var.op.name, var) for var in tf.trainable_variables()]
            self.summary_op = tf.summary.merge_all()

def visualize_clusters(X):
    model = sklearn.manifold.TSNE(n_components=2, random_state=0)
    tsne_enc = model.fit_transform(X)
    plt.scatter(tsne_enc[:,0], tsne_enc[:,1])
    plt.show()

def train_dec(name='dec'):
	# try to load previously trained weights
	weight_str = "-".join([str(x) for x in FLAGS.layer_sizes])
	checkpoint_dir = os.path.join(FLAGS.save_dir, '%s-%s' % (name, weight_str))
	check_save_dir(checkpoint_dir)

	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	data = mnist.train
	k_features = data.next_batch(1)[0].shape[1]
	encoded, init_weights, init_biases = stacked_ae(data)

    # run kmeans on the encoded data
	best_dist = -1
	for k in range(10,11):
		kmeans = sklearn.cluster.KMeans(n_clusters=k)
		kmeans.fit(encoded)
		labels_ = kmeans.predict(encoded)
		dist = sklearn.metrics.silhouette_score(encoded, labels_)
		if dist > best_dist:
			init_centriods = kmeans.cluster_centers_

	with tf.Graph().as_default():
		with tf.device("/gpu:0"):
			dec = DEC(init_weights, init_biases, init_centriods)

		saver = tf.train.Saver()
		init_op = tf.global_variables_initializer()
		with tf.Session() as sess:
			sess.run(init_op)
			_try_loading_checkpoint(checkpoint_dir, saver, sess)
			writer = tf.summary.FileWriter("./logs-dec/")
			writer.add_graph(tf.get_default_graph())
			def feed_dict(batch_size=256):
				return {dec.inputs: data.next_batch(batch_size)[0]}

			for i in range(FLAGS.dec_iters):
				out = sess.run([dec.KL, dec.optimize, dec.summary_op], feed_dict=feed_dict())
				if i % 10 == 0:
					print "(Training DEC) Iteration %i, KL: %2.5f" % (i, out[0])
					writer.add_summary(out[2], i)
					save_path = saver.save(sess, os.path.join(checkpoint_dir, "model_%08i.ckpt" % i))
			encoded = sess.run(dec.encoded, feed_dict=feed_dict(500))
		visualize_clusters(encoded)


if __name__ == "__main__":
    train_dec()
