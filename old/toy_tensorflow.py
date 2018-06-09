from tensorflow.python.framework import dtypes

from data import *

import argparse
import tensorflow as tf


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')


def convert_images_to_features(images):
    features = {}
    for key in range(row_num * col_num):
        features[str(key)] = list(map(lambda x: x[key], images))
    return features


def train_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    return dataset


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)

    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)
    return dataset


def main(argv):
    args = parser.parse_args(argv[1:])

    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in range(row_num * col_num):
        my_feature_columns.append(tf.feature_column.numeric_column(key="%d" % key, dtype=dtypes.uint8))
    print(my_feature_columns)

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    tf.estimator.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        hidden_units=[20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20],
        n_classes=10)

    # Train the Model.
    classifier.train(
        input_fn=lambda:train_input_fn(train_images_features, train_labels, args.batch_size),
        steps=args.train_steps)

    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda:eval_input_fn(test_images_features, test_labels, args.batch_size))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
