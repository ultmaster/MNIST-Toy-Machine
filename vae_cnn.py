import numpy as np
import tensorflow as tf

from vae import VariationalAutoEncoder

tf.logging.set_verbosity(tf.logging.INFO)


class ConvolutionalVariationalAutoEncoder(VariationalAutoEncoder):

    def __init__(self, network_architecture, learning_rate=0.001, batch_size=100):
        super().__init__(network_architecture, learning_rate=learning_rate, batch_size=batch_size)

    def _initialize_network(self, n_z, n_width, n_hidden_units, n_layers, **kwargs):
        self.n_z = n_z
        self.image_width = n_width
        self.n_hidden_units = n_hidden_units
        self.n_layers = n_layers
        self.input_layers = [None for i in range(self.n_layers + 1)]
        self.output_layers = [None for i in range(self.n_layers + 1)]

    def _recognition_network(self):
        self.input_layers[0] = tf.reshape(self.x, [-1, self.image_width, self.image_width, 1])
        self.ft_size = self.image_width // 2
        self.conv_size = self.image_width
        for i in range(1, self.n_layers + 1):
            self.conv_size //= 2
            self.ft_size *= 2
            self.input_layers[i] = tf.layers.conv2d(
                inputs=self.input_layers[i - 1],
                kernel_size=4,
                strides=2,
                filters=self.ft_size,
                padding="same",
                activation=tf.nn.relu,
            )
        self.z_mean = tf.layers.dense(
            inputs=tf.layers.flatten(self.input_layers[-1]),
            units=self.n_z,
            name="z_mean",
        )
        self.z_log_sigma_sq = tf.layers.dense(
            inputs=tf.layers.flatten(self.input_layers[-1]),
            units=self.n_z,
            name="z_log_sigma_sq",
        )

    def _generator_network(self):
        self.output_layers[0] = tf.reshape(tf.layers.dense(
            inputs=self.z,
            units=self.conv_size ** 2 * self.ft_size),
            (-1, self.conv_size, self.conv_size, self.ft_size))

        for i in range(1, self.n_layers + 1):
            self.ft_size //= 2
            self.output_layers[i] = tf.layers.conv2d_transpose(
                inputs=self.output_layers[i - 1],
                kernel_size=4,
                strides=2,
                filters=self.ft_size,
                padding="same",
                activation=tf.nn.relu,
            )
        self.x_reconstr_mean = tf.layers.flatten(tf.layers.conv2d(
            inputs=self.output_layers[-1],
            kernel_size=4,
            strides=1,
            filters=1,
            padding="same",
            activation=tf.nn.sigmoid,
        ))
