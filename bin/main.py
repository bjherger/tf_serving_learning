#!/usr/bin/env python
"""
coding=utf-8

Code template courtesy https://github.com/bjherger/Python-starter-repo

"""
import logging

import pandas
# TODO Clean up imports
import tensorflow as tf
from keras import backend as K, Sequential, Input, Model, losses, optimizers
from keras.layers import Dense, Concatenate
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.saved_model import builder as saved_model_builder, signature_constants, tag_constants

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

    dimm_1_shape = 1
    dimm_2_shape = 1

    # Load data
    logging.info('Loading data')

    dataframe = pandas.read_csv("iris.csv", header=None)
    observations = dataframe.values

    observations_x1 = observations[:, 0:dimm_1_shape].astype(float)
    observations_x2 = observations[:, dimm_1_shape:dimm_2_shape+1].astype(float)
    observations_y = observations[:, 4]
    logging.debug('observations_x1: {}'.format(observations_x1))
    logging.debug('observations_x2: {}'.format(observations_x2))
    logging.debug('observations_y: {}'.format(observations_y))

    # One hot encode response
    logging.info('OHE-ing response variable')
    encoder = LabelEncoder()
    encoder.fit(observations_y)
    encoded_Y = encoder.transform(observations_y)
    one_hot_labels = np_utils.to_categorical(encoded_Y)

    # Create model
    logging.info('Creating model')

    input_layer1 = Input(shape=(dimm_1_shape,), name='x1_input')
    input_layer2 = Input(shape=(dimm_2_shape,), name='x2_input')

    input_layers = [input_layer1, input_layer2]
    layers = Concatenate()(input_layers)
    layers = Dense(32)(layers)
    layers = Dense(3)(layers)

    model = Model(input_layers, layers)

    model.compile(loss=losses.categorical_crossentropy, optimizer='adam')

    model.fit([observations_x1, observations_x2], one_hot_labels)

    print(model.input)
    print(model.inputs)
    print(model.input_spec)

    # Create TF variable x1
    logging.info('Creating TF variable x1')
    serialized_tf_example = tf.placeholder(tf.string, name='x1')
    feature_configs = {'x1': tf.FixedLenFeature(shape=[dimm_1_shape], dtype=tf.float32), }
    tf_example = tf.parse_example(serialized_tf_example, feature_configs)
    x1 = tf.identity(tf_example['x1'], name='x1')  # use tf.identity() to assign name

    logging.info('Creating TF variable x2')
    serialized_tf_example = tf.placeholder(tf.string, name='x2')
    feature_configs = {'x2': tf.FixedLenFeature(shape=[dimm_2_shape], dtype=tf.float32), }
    tf_example = tf.parse_example(serialized_tf_example, feature_configs)
    x2 = tf.identity(tf_example['x2'], name='x2')  # use tf.identity() to assign name

    y = model([x1])
    logging.info('tf x: {}'.format(x1))
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
    # classification_inputs = tf.saved_model.utils.build_tensor_info(serialized_tf_example)
    # classification_outputs_classes = tf.saved_model.utils.build_tensor_info(prediction_classes)
    # classification_outputs_scores = tf.saved_model.utils.build_tensor_info(values)

    # Create keras signature
    classification_signature = tf.saved_model.signature_def_utils.classification_signature_def(
        examples=serialized_tf_example,
        classes=prediction_classes,
        scores=values
    )

    # Create tf_serving signature
    prediction_signature = tf.saved_model.signature_def_utils.predict_signature_def({"x1": x1}, {"prediction": y})

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
