import logging

import pandas
import tensorflow as tf
import keras.backend as K
from keras import Input, Model, losses
from keras.layers import Concatenate, Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.saved_model import signature_constants, signature_def_utils
from tensorflow.python.saved_model.signature_def_utils_impl import classification_signature_def, predict_signature_def
from tensorflow.python.saved_model.simple_save import simple_save

from bin import lib


def main():
    logging.getLogger().setLevel(logging.DEBUG)
    # Reference variable setup
    sess = tf.Session()
    K.set_session(sess)
    model_version = lib.get_batch_name()
    K.set_learning_phase(0)

    # Data setup

    logging.info('Loading data')

    dataframe = pandas.read_csv("iris_with_header.csv")
    numerical_cols = ['sepal_length', 'sepal_width']
    input_data = list()
    for col in numerical_cols:
        input_data.append(dataframe[col].values)

    logging.info('OHE-ing response variable')
    encoder = LabelEncoder()
    encoder.fit(dataframe.values[:, 4])
    encoded_Y = encoder.transform(dataframe.values[:, 4])
    one_hot_labels = np_utils.to_categorical(encoded_Y)

    # Model setup
    logging.info('Creating model')

    input_layers = list()

    for col in numerical_cols:
        logging.info('Creating input for {}'.format(col))

        if len(dataframe[col].shape) > 1:
            shape = dataframe[col].shape[1]
        else:
            shape = 1

        logging.info('Inferring variable {} has shape: {}'.format(col, shape))

        input_layers.append(Input(shape=(shape,), name='{}_input'.format(col)))

    layers = Concatenate()(input_layers)
    layers = Dense(32)(layers)
    layers = Dense(3)(layers)

    model = Model(input_layers, layers)

    model.compile(loss=losses.categorical_crossentropy, optimizer='adam')

    model.fit(input_data, one_hot_labels)

    tf_inputs = list()
    tf_examples = list()
    # Register input placholders
    for col in numerical_cols:
        logging.info('Creating tf placeholder for col: {}'.format(col))

        if len(dataframe[col].shape) > 1:
            shape = dataframe[col].shape[1]
        else:
            shape = 1
        logging.info('Inferring variable {} has shape: {}'.format(col, shape))

        serialized_tf_example = tf.placeholder(tf.string, name=col)
        tf_examples.append(serialized_tf_example)

        # TODO Better type lookup based on numpy types
        feature_configs = {col: tf.FixedLenFeature(shape=shape, dtype=tf.float32), }
        tf_example = tf.parse_example(serialized_tf_example, feature_configs)
        tf_inputs.append(tf.identity(tf_example[col], name=col))

    # Generate output tensor by feeding inputs into model
    y = model(tf_inputs)

    # Generate classification signature definition
    labels = encoder.classes_
    values, indices = tf.nn.top_k(y, len(labels))
    OHE_index_to_string_lookup_table = tf.contrib.lookup.index_to_string_table_from_tensor(tf.constant(labels))
    prediction_classes = OHE_index_to_string_lookup_table.lookup(tf.to_int64(indices))

    # Save model
    output_path = './' + model_version
    logging.info('Saving model to {}'.format(output_path))
    simple_save_inputs = dict(zip(numerical_cols, tf_inputs))
    logging.info('Inputs to simple save: {}'.format(simple_save_inputs))
    simple_save(sess, output_path, inputs=simple_save_inputs, outputs={'y': y})

    # Can now get signature w/ https://www.tensorflow.org/guide/saved_model#cli_to_inspect_and_execute_savedmodel

    # TODO Create tensorflow-serving signature

    # TODO validate signatures

    # TODO Serialize tensorflow-serving elements

    # TODO Save the graph

    # TODO Return path to serialized graph

    pass


if __name__ == '__main__':
    main()
