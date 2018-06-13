from datetime import datetime

import numpy as np
import tensorflow as tf

from vae import VariationalAutoEncoder

tf.logging.set_verbosity(tf.logging.INFO)


class CVAE(VariationalAutoEncoder):

    def __init__(self, network_architecture, learning_rate=0.001, batch_size=100):
        self.y = tf.placeholder(tf.int64, shape=(None,), name="y")
        super().__init__(network_architecture, learning_rate=learning_rate, batch_size=batch_size)

    def _initialize_network(self, n_z, n_width, n_hidden_units, n_layers, **kwargs):
        self.n_z = n_z
        self.image_width = n_width
        self.n_hidden_units = n_hidden_units
        self.n_layers = n_layers
        self.input_layers = [None for i in range(self.n_layers + 2)]
        self.output_layers = [None for i in range(self.n_layers + 2)]

    def _create_network(self):
        with tf.name_scope("labels"):
            self.input_label = tf.feature_column.input_layer(
                {"label": self.y},
                [tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_identity(
                    key='label',
                    num_buckets=10))]
            )
        super()._create_network()

    def _recognition_network(self):
        # Generate probabilistic encoder (recognition network), which
        # maps inputs onto a normal distribution in latent space.
        # The transformation is parametrized and can be learned.
        self.input_layers[0] = tf.reshape(self.x, [-1, self.image_width, self.image_width, 1])
        self.ft_size = 8
        self.conv_size = self.image_width
        for i in range(1, self.n_layers + 1):
            self.conv_size //= 2
            self.ft_size *= 2
            self.input_layers[i] = tf.layers.conv2d(
                inputs=self.input_layers[i - 1],
                kernel_size=3,
                strides=2,
                filters=self.ft_size,
                padding="same",
                activation=tf.nn.softplus,
            )
        self.dense_with_label = tf.concat([tf.layers.flatten(self.input_layers[self.n_layers]),
                                           self.input_label], 1)
        self.z_mean = tf.layers.dense(
            inputs=self.dense_with_label,
            units=self.n_z,
            activation=None,
            name="z_mean",
        )
        self.z_log_sigma_sq = tf.layers.dense(
            inputs=self.dense_with_label,
            units=self.n_z,
            activation=None,
            name="z_log_sigma_sq",
        )

    def _generator_network(self):
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto a Bernoulli distribution in data space.
        # The transformation is parametrized and can be learned.
        self.output_layers[0] = tf.reshape(tf.layers.dense(
            inputs=tf.concat([self.z, self.input_label], 1),
            units=self.conv_size ** 2 * self.ft_size),
            (-1, self.conv_size, self.conv_size, self.ft_size))

        for i in range(1, self.n_layers + 1):
            self.ft_size //= 2
            self.output_layers[i] = tf.layers.conv2d_transpose(
                inputs=self.output_layers[i - 1],
                kernel_size=3,
                strides=2,
                filters=self.ft_size,
                padding="same",
                activation=tf.nn.softplus,
            )
        self.x_reconstr_mean = tf.layers.flatten(tf.layers.conv2d(
            inputs=self.output_layers[self.n_layers],
            kernel_size=2,
            strides=1,
            filters=1,
            padding="same",
            activation=tf.nn.sigmoid,
        ))

    def partial_fit_with_y(self, X, y):
        """
        Train model based on mini-batch of input data.
        Return cost of mini-batch.
        """
        opt, cost = self.sess.run((self.optimizer, self.cost), feed_dict={self.x: X,
                                                                          self.y: y,
                                                                          self.training: True})
        return cost

    def transform_with_y(self, X, y):
        """
        Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.z_mean, feed_dict={self.x: X, self.y: y, self.training: False})

    def generate_with_y(self, y, z_mu=None):
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
                             feed_dict={self.z: z_mu, self.y: y, self.training: False})

    def reconstruct_with_y(self, X, y):
        """ Use VAE to reconstruct given data. """
        return self.sess.run(self.x_reconstr_mean,
                             feed_dict={self.x: X, self.y: y, self.training: False})

    def reconstruct_error(self, X, y):
        return self.sess.run(self.reconstruct_loss,
                             feed_dict={self.x: X, self.y: y, self.training: False})

    def predict(self, X):
        med = np.zeros((X.shape[0], 10), dtype=np.float32)
        for clas in range(10):
            guess = np.asarray([clas] * X.shape[0])
            med[:, clas] = self.reconstruct_error(X, guess)
        return np.argmin(med, axis=1)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.sum(y == y_pred) / X.shape[0]


import mnist
import time

def train_cvae(network_architecture, learning_rate=0.001,
          batch_size=100, training_epochs=10, display_step=1, saving_step=10):
    model_name = "CVAE." + ''.join(filter(lambda x: x.isdigit(), datetime.now().isoformat()))
    vae = CVAE(network_architecture,
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
            batch_ys = train_y[i:i + batch_size]

            cost = vae.partial_fit_with_y(batch_xs, batch_ys)
            avg_cost += cost / n_samples * batch_size
        t1 = time.time()

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1),
                  "Cost:", "{:.9f}".format(avg_cost),
                  "Time:", "%.3f" % (t1 - t0))
        if epoch % saving_step == 0:
            test_x, test_y = mnist.test_32_flat_labeled()
            print("Accuracy:", vae.score(test_x, test_y))
            vae.save("tmp/" + model_name + ".step%d" % epoch)

    return vae

network_architecture = {
    "n_width": 32,
    "n_z": 20,  # dimensionality of latent space
    "n_hidden_units": 500,
    "n_layers": 3,
}

vae = train_cvae(network_architecture, saving_step=5, training_epochs=200)