import logging

import pandas
import tensorflow as tf
import keras.backend as K
from keras import Input, Model, losses
from keras.layers import Concatenate, Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

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

    # Register input placholders
    # for

    # TODO Generate output tensor by feeding inputs into model

    # TODO Generate output label mapping

    # TODO Generate classification signature definition

    # TODO Create tensorflow-serving signature

    # TODO validate signatures

    # TODO Serialize tensorflow-serving elements

    # TODO Save the graph

    # TODO Return path to serialized graph

    pass

if __name__ == '__main__':
    main()