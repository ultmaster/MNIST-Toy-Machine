import numpy as np
import tensorflow as tf
from data import *

tf.logging.set_verbosity(tf.logging.INFO)


class ConvolutionalConditionalVariationalAutoEncoder(object):

    def __init__(self, network_architecture, transfer_fct=tf.nn.softplus,
                 learning_rate=0.001, batch_size=100):
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # tf Graph input
        self.x = tf.placeholder(tf.float32, [None, network_architecture["n_input"]])
        self.y = tf.placeholder(tf.uint8, [None, 1])

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

    def _create_network(self):
        self._initialize_weights(**self.network_architecture)

        self.input_label = tf.feature_column.input_layer(
            {"label": self.y},
            [tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_identity(
                key='label',
                num_buckets=10))]
        )

        self._recognition_network()

        eps = tf.random_normal((self.batch_size, self.n_z), 0, 1,
                               dtype=tf.float32)
        # z = mu + sigma*epsilon
        self.z = tf.add(self.z_mean,
                        tf.multiply(tf.exp(self.z_log_sigma_sq / 2), eps))
        self._generator_network()

    def _initialize_weights(self, n_z, **kwargs):
        self.n_z = n_z

    def _recognition_network(self):
        self.input_layer = tf.reshape(self.x, [-1, 28, 28, 1])
        self.conv1 = tf.layers.conv2d(
            inputs=self.input_layer,
            filters=32,
            kernel_size=3,
            strides=2,
            padding="same",
            activation=tf.nn.relu
        )
        self.conv2 = tf.layers.conv2d(
            inputs=self.conv1,
            filters=64,
            kernel_size=3,
            strides=2,
            padding="same",
            activation=tf.nn.relu
        )
        self.conv2_flat = tf.reshape(self.conv2, [-1, 7 * 7 * 64])
        self.dense1 = tf.layers.dense(
            inputs=self.conv2_flat,
            units=20,
            activation=tf.nn.relu
        )
        self.dense_with_label = tf.concat([self.dense1, self.input_label], 1)
        # self.dense_with_label = tf.Print(self.dense_with_label, [tf.shape(self.dense_with_label)])
        self.z_mean = tf.layers.dense(
            inputs=self.dense_with_label,
            units=self.n_z,
            activation=tf.nn.relu
        )
        self.z_log_sigma_sq = tf.layers.dense(
            inputs=self.dense_with_label,
            units=self.n_z,
            activation=tf.nn.relu
        )

    def _generator_network(self):
        self.z_with_label = tf.concat([self.z, self.input_label], 1)
        self.dense3 = tf.layers.dense(
            inputs=self.z_with_label,
            units=7 * 7 * 64,
            activation=tf.nn.relu
        )
        self.dense2_reshape = tf.reshape(self.dense3, [-1, 7, 7, 64])
        self.dconv1 = tf.layers.conv2d_transpose(
            inputs=self.dense2_reshape,
            filters=64,
            kernel_size=3,
            strides=2,
            padding="same",
            activation=tf.nn.relu)
        self.dconv2 = tf.layers.conv2d_transpose(
            inputs=self.dconv1,
            kernel_size=3,
            filters=32,
            strides=2,
            padding="same",
            activation=tf.nn.relu)
        self.dconv3 = tf.layers.conv2d_transpose(
            inputs=self.dconv2,
            kernel_size=3,
            filters=1,
            strides=1,
            padding="same",
            activation=tf.nn.sigmoid)
        self.x_reconstr_mean = tf.reshape(self.dconv3, [-1, 784])

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
        #     between the distribution in latent space induced by the encoder on
        #     the data and some prior. This acts as a kind of regularizer.
        #     This can be interpreted as the number of "nats" required
        #     for transmitting the the latent space distribution given
        #     the prior.
        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq
                                           - tf.square(self.z_mean)
                                           - tf.exp(self.z_log_sigma_sq), 1)

        self.cost = tf.reduce_mean(reconstr_loss + latent_loss)  # average over batch
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    def partial_fit(self, X, y):
        """Train model based on mini-batch of input data.

        Return cost of mini-batch.
        """
        opt, cost = self.sess.run((self.optimizer, self.cost),
                                  feed_dict={self.x: X, self.y: y.reshape([-1, 1])})
        return cost

    def transform(self, X, y):
        """Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.z_mean, feed_dict={self.x: X, self.y: y.reshape([-1, 1])})

    def generate(self, y, z_mu=None):
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
                             feed_dict={self.z: z_mu, self.y: y.reshape([-1, 1])})

    def reconstruct(self, X, y):
        """ Use VAE to reconstruct given data. """
        return self.sess.run(self.x_reconstr_mean,
                             feed_dict={self.x: X, self.y: y.reshape([-1, 1])})


def train(network_architecture, learning_rate=0.001,
          batch_size=100, training_epochs=10, display_step=1):
    vae = ConvolutionalConditionalVariationalAutoEncoder(network_architecture,
                                                         learning_rate=learning_rate,
                                                         batch_size=batch_size)
    # Training cycle
    n_samples = len(train_images)
    for epoch in range(training_epochs):
        avg_cost = 0.
        # Loop over all batches
        for i in range(0, n_samples, batch_size):
            batch_features = np.asarray(train_images[i:i + batch_size], dtype=np.float32) / 256
            batch_label = np.asarray(train_labels[i:i + batch_size], dtype=np.uint8)

            # Fit training using batch data
            cost = vae.partial_fit(batch_features, batch_label)
            # Compute average loss
            avg_cost += cost / n_samples * batch_size

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1),
                  "cost=", "{:.9f}".format(avg_cost))
    return vae


network_architecture = {
    "n_input": 784,  # MNIST data input (img shape: 28*28)
    "n_z": 2,  # dimensionality of latent space
}

cvae_2d = train(network_architecture, training_epochs=30)
