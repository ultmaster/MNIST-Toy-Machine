from datetime import datetime

import numpy as np
import tensorflow as tf
import time

import mnist

tf.logging.set_verbosity(tf.logging.INFO)


class VariationalAutoEncoder(object):

    def __init__(self, network_architecture, learning_rate=0.001, batch_size=100):
        self.network_architecture = network_architecture
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # tf Graph input
        self._initialize_network(**self.network_architecture)
        self.x = tf.placeholder(tf.float32, shape=[None, self.image_width * self.image_width], name="input")
        self.training = tf.placeholder(tf.bool, shape=(), name='apply_dropout')

        # Create autoencoder network
        self._create_network()
        # Define loss function based variational upper-bound and
        # corresponding optimizer
        self._create_loss_optimizer()

        # Initializing the tensor flow variables
        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()  # to save models

        # Launch the session
        self.sess = tf.InteractiveSession()
        self.writer = tf.summary.FileWriter(".", self.sess.graph)
        self.sess.run(init)

    def _create_network(self):
        # Use recognition network to determine mean and
        # (log) variance of Gaussian distribution in latent space
        self._recognition_network()

        # Draw one sample z from Gaussian distribution
        eps = tf.random_normal((tf.shape(self.x)[0], self.n_z), 0, 1, dtype=tf.float32)
        # z = mu + sigma * epsilon
        self.z = tf.add(self.z_mean, tf.multiply(tf.exp(self.z_log_sigma_sq / 2), eps))

        # Use generator to determine mean of
        # Bernoulli distribution of reconstructed input
        self._generator_network()

    def _initialize_network(self, n_z, n_width, n_hidden_units, n_layers, **kwargs):
        self.n_z = n_z
        self.image_width = n_width
        self.n_hidden_units = n_hidden_units
        self.n_layers = n_layers
        self.input_layers = [None for i in range(self.n_layers + 1)]
        self.output_layers = [None for i in range(self.n_layers + 1)]

    def _recognition_network(self):
        # Generate probabilistic encoder (recognition network), which
        # maps inputs onto a normal distribution in latent space.
        # The transformation is parametrized and can be learned.
        self.input_layers[0] = tf.layers.flatten(self.x)
        for i in range(1, self.n_layers + 1):
            self.input_layers[i] = tf.layers.dense(
                inputs=self.input_layers[i - 1],
                units=self.n_hidden_units,
                activation=tf.nn.softplus,
            )
        # self.recog_dropout_layer = tf.layers.dropout(
        #     inputs=self.input_layers[-1],
        #     rate=0,
        #     training=self.training,
        # )
        self.z_mean = tf.layers.dense(
            inputs=self.input_layers[-1],
            units=self.n_z,
            activation=None,
            name="z_mean",
        )
        self.z_log_sigma_sq = tf.layers.dense(
            inputs=self.input_layers[-1],
            units=self.n_z,
            activation=None,
            name="z_log_sigma_sq",
        )

    def _generator_network(self):
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto a Bernoulli distribution in data space.
        # The transformation is parametrized and can be learned.
        self.output_layers[0] = self.z
        for i in range(1, self.n_layers + 1):
            self.output_layers[i] = tf.layers.dense(
                inputs=self.output_layers[i - 1],
                units=self.n_hidden_units,
                activation=tf.nn.softplus,
            )
        self.x_reconstr_mean = tf.layers.dense(
            inputs=self.output_layers[-1],
            units=self.image_width * self.image_width,
            activation=tf.nn.sigmoid,
        )

    def _reconstruct_loss(self):
        self.reconstruct_loss = -tf.reduce_sum(self.x * tf.log(1e-10 + self.x_reconstr_mean) +
                                               (1 - self.x) * tf.log(1e-10 + 1 - self.x_reconstr_mean), 1)
        return self.reconstruct_loss

    def _create_loss_optimizer(self):
        with tf.name_scope("cost"):
            # The loss is composed of two terms:
            # 1.) The reconstruction loss (the negative log probability
            #     of the input under the reconstructed Bernoulli distribution
            #     induced by the decoder in the data space).
            #     This can be interpreted as the number of "nats" required
            #     for reconstructing the input when the activation in latent
            #     is given.
            # Adding 1e-10 to avoid evaluation of log(0.0)
            reconstr_loss = self._reconstruct_loss()

            # 2.) The latent loss, which is defined as the KL divergence
            #     between the distribution in latent space induced by the encoder on
            #     the data and some prior. This acts as a kind of regularizer.
            #     This can be interpreted as the number of "nats" required
            #     for transmitting the the latent space distribution given
            #     the prior.
            latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq - tf.square(self.z_mean)
                                               - tf.exp(self.z_log_sigma_sq), 1)

            self.cost = tf.reduce_mean(reconstr_loss + latent_loss, name="total_cost")  # average over batch
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    def partial_fit(self, X):
        """
        Train model based on mini-batch of input data.
        Return cost of mini-batch.
        """
        opt, cost = self.sess.run((self.optimizer, self.cost), feed_dict={self.x: X, self.training: True})
        return cost

    def transform(self, X):
        """
        Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.z_mean, feed_dict={self.x: X, self.training: False})

    def generate(self, z_mu=None):
        """
        Generate data by sampling from latent space.
        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent
        space.
        """
        if z_mu is None:
            z_mu = np.random.normal(size=self.network_architecture["n_z"])
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.x_reconstr_mean,
                             feed_dict={self.z: z_mu, self.training: False})

    def reconstruct(self, X):
        """ Use VAE to reconstruct given data. """
        return self.sess.run(self.x_reconstr_mean,
                             feed_dict={self.x: X, self.training: False})

    def save(self, path):
        save_path = self.saver.save(self.sess, path)
        print("Model has been saved to: %s" % save_path)

    def load(self, path):
        self.saver.restore(self.sess, path)


def train(cls, network_architecture, learning_rate=0.001,
          batch_size=100, training_epochs=10, display_step=1, saving_step=10):
    model_name = cls.__name__ + "." + ''.join(filter(lambda x: x.isdigit(), datetime.now().isoformat()))

    vae = cls(network_architecture,
              learning_rate=learning_rate,
              batch_size=batch_size)
    # Training cycle
    train_x, train_y = mnist.train_32_flat_labeled()
    n_samples = train_x.shape[0]
    w = network_architecture["n_width"]

    t0 = time.time()
    for epoch in range(training_epochs):
        avg_cost = 0.
        # Loop over all batches
        for i in range(0, n_samples, batch_size):
            batch_xs = train_x[i:i + batch_size].reshape((-1, w * w))
            batch_xs = batch_xs.reshape((-1, w * w))

            cost = vae.partial_fit(batch_xs)
            avg_cost += cost / n_samples * batch_size
        t1 = time.time()

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1),
                  "Cost:", "{:.9f}".format(avg_cost),
                  "Time:", "%.3f" % (t1 - t0))
        if epoch % saving_step == 0:
            vae.save("tmp/" + model_name + ".step%d" % epoch)

    return vae
