import numpy as np
import tensorflow as tf
from data import *


tf.logging.set_verbosity(tf.logging.INFO)


def xavier_init(fan_in, fan_out, constant=1):
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)


class VariationalAutoencoder(object):
    """ Variation Autoencoder (VAE) with an sklearn-like interface implemented using TensorFlow.

    This implementation uses probabilistic encoders and decoders using Gaussian
    distributions and  realized by multi-layer perceptrons. The VAE can be learned
    end-to-end.

    See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.
    """

    def __init__(self, network_architecture, transfer_fct=tf.nn.softplus,
                 learning_rate=0.001, batch_size=100):
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # tf Graph input
        self.x = tf.placeholder(tf.float32, [None, network_architecture["n_input"]])

        # Create autoencoder network
        self._create_network()
        # Define loss function based variational upper-bound and
        # corresponding optimizer
        self._create_loss_optimizer()

        # Initializing the tensor flow variables
        init = tf.global_variables_initializer()

        # Launch the session
        self.sess = tf.InteractiveSession()
        self.sess.run(init)
        writer = tf.summary.FileWriter('.')
        writer.add_graph(tf.get_default_graph())

    def _create_network(self):
        # Initialize autoencode network weights and biases
        network_weights = self._initialize_weights(**self.network_architecture)

        # Use recognition network to determine mean and
        # (log) variance of Gaussian distribution in latent
        # space
        self._recognition_network()

        # Draw one sample z from Gaussian distribution
        eps = tf.random_normal((self.batch_size, self.n_z), 0, 1,
                               dtype=tf.float32)
        # z = mu + sigma*epsilon
        self.z = tf.add(self.z_mean,
                        tf.multiply(tf.exp(self.z_log_sigma_sq / 2), eps))

        # Use generator to determine mean of
        # Bernoulli distribution of reconstructed input
        self._generator_network()

    def _initialize_weights(self, n_z, **kwargs):
        self.n_z = n_z

    def _recognition_network(self):
        # Generate probabilistic encoder (recognition network), which
        # maps inputs onto a normal distribution in latent space.
        # The transformation is parametrized and can be learned.
        self.input_layer = tf.reshape(self.x, [-1, 28, 28, 1])
        self.conv1 = tf.layers.conv2d(
            inputs=self.input_layer,
            filters=32,
            kernel_size=[10, 10],
            padding="valid",
            activation=tf.nn.relu
        )
        self.conv2 = tf.layers.conv2d(
            inputs=self.conv1,
            filters=64,
            kernel_size=[10, 10],
            padding="valid",
            activation=tf.nn.relu
        )
        self.conv2_flat = tf.reshape(self.conv2, [-1, 10 * 10 * 64])
        self.dense1 = tf.layers.dense(
            inputs=self.conv2_flat,
            units=1024,
            activation=tf.nn.relu
        )
        self.z_mean = tf.layers.dense(
            inputs=self.dense1,
            units=self.n_z,
            activation=tf.nn.relu
        )
        self.z_log_sigma_sq = tf.layers.dense(
            inputs=self.dense1,
            units=self.n_z,
            activation=tf.nn.relu
        )

    # def _sample_z(self):
    #     noise = tf.random_normal(shape=(self.batch_size, self.n_z), mean=0., std=1.)
    #     return tf.add(self.z_mean, tf.matmul(tf.exp(self.z_log_sigma_sq / 2) * noise))

    def _generator_network(self):
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto a Bernoulli distribution in data space.
        # The transformation is parametrized and can be learned.

        self.dense3 = tf.layers.dense(
            inputs=self.z,
            units=10 * 10 * 64,
            activation=tf.nn.relu
        )
        self.dense2_reshape = tf.reshape(self.dense3, [-1, 10, 10, 64])
        self.dconv1 = tf.layers.conv2d_transpose(
            inputs=self.dense2_reshape,
            filters=32,
            kernel_size=[10, 10],
            padding="valid",
            activation=tf.nn.relu)
        self.dconv2 = tf.layers.conv2d_transpose(
            inputs=self.dconv1,
            filters=1,
            kernel_size=[10, 10],
            padding="valid",
            activation=tf.nn.relu)
        self.x_reconstr_mean = tf.reshape(self.dconv2, [-1, 784])

    def _create_loss_optimizer(self):
        # The loss is composed of two terms:
        # 1.) The reconstruction loss (the negative log probability
        #     of the input under the reconstructed Bernoulli distribution
        #     induced by the decoder in the data space).
        #     This can be interpreted as the number of "nats" required
        #     for reconstructing the input when the activation in latent
        #     is given.
        # Adding 1e-10 to avoid evaluation of log(0.0)
        reconstr_loss = \
            -tf.reduce_sum(self.x * tf.log(1e-10 + self.x_reconstr_mean)
                           + (1 - self.x) * tf.log(1e-10 + 1 - self.x_reconstr_mean),
                           1)
        # 2.) The latent loss, which is defined as the Kullback Leibler divergence
        ##    between the distribution in latent space induced by the encoder on
        #     the data and some prior. This acts as a kind of regularizer.
        #     This can be interpreted as the number of "nats" required
        #     for transmitting the the latent space distribution given
        #     the prior.
        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq
                                           - tf.square(self.z_mean)
                                           - tf.exp(self.z_log_sigma_sq), 1)
        self.cost = tf.reduce_mean(reconstr_loss + latent_loss)  # average over batch
        # Use ADAM optimizer
        self.optimizer = \
            tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    def partial_fit(self, X):
        """Train model based on mini-batch of input data.

        Return cost of mini-batch.
        """
        opt, cost = self.sess.run((self.optimizer, self.cost),
                                  feed_dict={self.x: X})
        return cost

    def transform(self, X):
        """Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.z_mean, feed_dict={self.x: X})

    def generate(self, z_mu=None):
        """ Generate data by sampling from latent space.

        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent
        space.
        """
        if z_mu is None:
            z_mu = np.random.normal(size=self.network_architecture["n_z"])
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.x_reconstr_mean,
                             feed_dict={self.z: z_mu})

    def reconstruct(self, X):
        """ Use VAE to reconstruct given data. """
        return self.sess.run(self.x_reconstr_mean,
                             feed_dict={self.x: X})


def train(network_architecture, learning_rate=0.001,
          batch_size=100, training_epochs=10, display_step=5):
    vae = VariationalAutoencoder(network_architecture,
                                 learning_rate=learning_rate,
                                 batch_size=batch_size)
    # Training cycle
    n_samples = len(train_images)
    for epoch in range(training_epochs):
        avg_cost = 0.
        # Loop over all batches
        for i in range(0, n_samples, batch_size):
            batch_xs = np.asarray(train_images[i:i + batch_size], dtype=np.float32) / 256

            # Fit training using batch data
            cost = vae.partial_fit(batch_xs)
            # Compute average loss
            avg_cost += cost / n_samples * batch_size

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(avg_cost))
    return vae


network_architecture = {
    "n_input": 784, # MNIST data input (img shape: 28*28)
    "n_z": 2, # dimensionality of latent space
}

vae = train(network_architecture, training_epochs=75)
