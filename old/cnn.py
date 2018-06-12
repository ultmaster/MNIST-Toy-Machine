import argparse

import mnist
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--central', dest='central', default=0, type=int, help='centralize')


def cnn_model_fn(features, labels, mode, params):
    """Model function for CNN."""
    # Input Layer
    width = params["width"]
    input_layer = tf.reshape(features["x"], [-1, width, width, 1])

    # Convolutional Layer
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    pool2_shape = pool2.get_shape()

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, pool2_shape[1] * pool2_shape[2] * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
    }
    # Generate accuracy during trainings
    predict_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(labels, tf.int64), predictions['classes']), tf.float32))

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        tensors_to_log = {
            'accuracy': predict_accuracy
        }
        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, training_hooks=[logging_hook])

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(argv):
    args = parser.parse_args(argv[1:])
    width = 45
    train_x, train_y = mnist.train()
    test_x, test_y = mnist.test()

    if args.central:
        print("Use centralize now")
        width = 32
        train_x = mnist.train_32()
        test_x = mnist.test_32()

    train_x = train_x / 256
    test_x = test_x / 256

    train_y = train_y.astype(np.int32)
    test_y = test_y.astype(np.int32)
    mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, params={"width": width})
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_x},
        y=train_y,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=10000)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_x},
        y=test_y,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
    tf.app.run()
