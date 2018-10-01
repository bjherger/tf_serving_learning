#!/usr/bin/env python
"""
coding=utf-8

Code template courtesy https://github.com/bjherger/Python-starter-repo

"""
import logging

import pandas
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.datasets import load_iris
from tensorflow.python.saved_model import builder as saved_model_builder, signature_constants, tag_constants

# TODO Clean up imports
import tensorflow as tf
from keras import backend as K, Sequential
from sklearn.preprocessing import LabelEncoder

from bin import lib


def main():
    """
    Main function documentation template
    :return: None
    :rtype: None
    """
    logging.basicConfig(level=logging.DEBUG)

    # Reference variables
    logging.info('Setting up reference variables')
    sess = tf.Session()
    K.set_session(sess)
    model_version = lib.get_batch_name()
    K.set_learning_phase(0)

    # Load data
    logging.info('Loading data')

    dataframe = pandas.read_csv("iris.csv", header=None)
    observations = dataframe.values
    observations_x = observations[:, 0:4].astype(float)
    observations_y = observations[:, 4]
    logging.debug('observations_x: {}'.format(observations_x))
    logging.debug('observations_y: {}'.format(observations_y))

    # One hot encode response
    logging.info('OHE-ing response variable')
    encoder = LabelEncoder()
    encoder.fit(observations_y)
    encoded_Y = encoder.transform(observations_y)
    one_hot_labels = np_utils.to_categorical(encoded_Y)

    # Create model
    logging.info('Creating model')
    # TODO Switch to keras's functional API
    model = Sequential()
    model.add(Dense(8, input_dim=4, activation="relu"))
    model.add(Dense(3, activation="softmax"))

    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    model.fit(observations_x, one_hot_labels, batch_size=32)

    # Create TF variables
    logging.info('Creating TF variables')
    serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
    feature_configs = {'x': tf.FixedLenFeature(shape=[4], dtype=tf.float32), }
    tf_example = tf.parse_example(serialized_tf_example, feature_configs)

    x = tf.identity(tf_example['x'], name='x')  # use tf.identity() to assign name
    y = model(x)
    logging.info('tf x: {}'.format(x))
    logging.info('tf y: {}'.format(y))

    # Build lookup table from prediction index to correct (string) label
    labels = []
    for label in observations_y:
        if label not in labels:
            labels.append(label)
    logging.info('Num labels: {}'.format(len(labels)))
    logging.debug('labels: {}'.format(labels))

    values, indices = tf.nn.top_k(y, len(labels))
    OHE_index_to_string_lookup_table = tf.contrib.lookup.index_to_string_table_from_tensor(tf.constant(labels))
    prediction_classes = OHE_index_to_string_lookup_table.lookup(tf.to_int64(indices))

    # Create tf variables
    classification_inputs = tf.saved_model.utils.build_tensor_info(serialized_tf_example)
    classification_outputs_classes = tf.saved_model.utils.build_tensor_info(prediction_classes)
    classification_outputs_scores = tf.saved_model.utils.build_tensor_info(values)

    # Create keras signature
    classification_signature = tf.saved_model.signature_def_utils.classification_signature_def(
        examples=serialized_tf_example,
        classes=prediction_classes,
        scores=values
    )

    # Create tf_serving signature
    prediction_signature = tf.saved_model.signature_def_utils.predict_signature_def({"inputs": x}, {"prediction": y})

    # Validate tf_serving signature
    valid_prediction_signature = tf.saved_model.signature_def_utils.is_valid_signature(prediction_signature)
    valid_classification_signature = tf.saved_model.signature_def_utils.is_valid_signature(classification_signature)

    if (valid_prediction_signature == False):
        raise ValueError("Error: Prediction signature not valid!")

    if (valid_classification_signature == False):
        raise ValueError("Error: Classification signature not valid!")

    # Serialize tf_serving elements
    builder = saved_model_builder.SavedModelBuilder('./' + model_version)
    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

    # Add the meta_graph and the variables to the builder
    builder.add_meta_graph_and_variables(
        sess, [tag_constants.SERVING],
        signature_def_map={
            'predict-iris':
                prediction_signature,
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                classification_signature,
        },
        legacy_init_op=legacy_init_op)

    # Save the graph
    save_path = builder.save()

    return save_path

# Main section
if __name__ == '__main__':
    main()
