import sys
import tensorflow as tf
import numpy as np
import mnist
import time


class ConvolutionNet:
    def __init__(self, image_width):
        tf.reset_default_graph()
        self.image_width = image_width

        # input layer
        with tf.name_scope("input"):
            self.x = tf.placeholder(tf.float32, shape=(None, self.image_width * self.image_width), name="x")
            self.y = tf.placeholder(tf.int64, shape=(None,), name="y")
            self.y_ = tf.feature_column.input_layer(
                {"label": self.y},
                [tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_identity(
                    key='label',
                    num_buckets=10))]
            )

        self.x_as_image = tf.reshape(self.x, [-1, self.image_width, self.image_width, 1])

        # First Convolution Layer
        with tf.name_scope("conv1"):
            self.conv1 = tf.layers.conv2d(
                inputs=self.x_as_image,
                kernel_size=5, strides=1, filters=32,
                padding="same", activation=tf.nn.relu,
                name="conv1"
            )
            self.pool1 = tf.layers.max_pooling2d(inputs=self.conv1, pool_size=2, strides=2, name="pool1")

        with tf.name_scope("conv2"):
            self.conv2 = tf.layers.conv2d(
                inputs=self.pool1,
                kernel_size=5, strides=1, filters=64,
                padding="same", activation=tf.nn.relu,
                name="conv2"
            )
            self.pool2 = tf.layers.max_pooling2d(inputs=self.conv2, pool_size=2, strides=2, name="pool2")

        with tf.name_scope("dense1"):
            self.dense1 = tf.layers.dense(
                inputs=tf.layers.flatten(self.pool2),
                units=1024, activation=tf.nn.relu,
                name="dense_layer"
            )

        with tf.name_scope("logits"):
            self.logits = tf.layers.dense(
                inputs=self.dense1,
                units=10,
                name="logits"
            )

        with tf.name_scope("cost"):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y_))
            self.cost = tf.reduce_mean(self.loss)
            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y_, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.sess = tf.Session()
        self.writer = tf.summary.FileWriter(".", self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def partial_fit(self, x, y):
        opt, cost = self.sess.run((self.optimizer, self.cost), feed_dict={self.x: x, self.y: y})
        return cost

    def score(self, x, y):
        return self.sess.run(self.accuracy, feed_dict={self.x: x, self.y: y})

    def save(self, path):
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, path)
        print("Model has been saved to %s" % save_path)

    def load(self, path):
        saver = tf.train.Saver()
        saver.restore(self.sess, tf.train.latest_checkpoint(path))


def main(training_size=None, max_epochs=20, save=True, display_step=1):
    width = 32
    cnn = ConvolutionNet(width)
    batch_size = 100

    # Training cycle
    train_x, train_y = mnist.train_32_flat_labeled(training_size)
    test_x, test_y = mnist.test_32_flat_labeled()
    n_samples = train_x.shape[0]
    best_acc, best_timeout = 0, 2

    t0 = time.time()
    for epoch in range(max_epochs):
        avg_cost = 0.
        # Loop over all batches
        for i in range(0, n_samples, batch_size):
            # print a progress bar
            sys.stdout.write('\r')
            sys.stdout.write("Epoch %03d [%-50s] %.3f%%" % (epoch + 1, '=' * int(i / n_samples * 50), (i / n_samples * 100)))
            sys.stdout.flush()

            batch_xs = train_x[i:i + batch_size]
            batch_ys = train_y[i:i + batch_size]

            cost = cnn.partial_fit(batch_xs, batch_ys)
            avg_cost += cost / n_samples * batch_size

        t1 = time.time()
        print()

        # Display logs per epoch step
        print("Cost:", "%.9f" % avg_cost,
              "Time:", "%.3f" % (t1 - t0))

        if (epoch + 1) % display_step == 0:
            accuracy = cnn.score(test_x, test_y)

            if accuracy > best_acc:
                best_acc = accuracy
                best_timeout = 2
            else:
                best_timeout -= 1
                if best_timeout <= 0:
                    print("Accuracy hasn't improved in last 2 steps. Quitting...")
                    break
            print("Accuracy:", accuracy, "Best:", best_acc)

        if save:
            cnn.save("tmp/cnn2.%d" % epoch)


if __name__ == "__main__":
    main(save=False, display_step=5)
    # for n_samples in [1000, 2000, 4000, 8000, 16000, 32000, None]:
    #     print("N_SAMPLES:", n_samples)
    #     main(training_size=n_samples, save=False)

