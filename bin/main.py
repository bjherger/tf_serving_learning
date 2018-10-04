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
    dimm_1_shape = 1
    dimm_2_shape = 1

    dataframe = pandas.read_csv("iris.csv", header=None)
    observations = dataframe.values

    observations_x1 = observations[:, 0:dimm_1_shape].astype(float)
    observations_x2 = observations[:, dimm_1_shape:dimm_2_shape + 1].astype(float)
    observations_y = observations[:, 4]
    input_data = [observations_x1, observations_x2]

    logging.info('OHE-ing response variable')
    encoder = LabelEncoder()
    encoder.fit(observations_y)
    encoded_Y = encoder.transform(observations_y)
    one_hot_labels = np_utils.to_categorical(encoded_Y)

    # TODO Model setup
    # Create model
    logging.info('Creating model')

    input_layers = list()

    for index, data in enumerate(input_data):
        logging.info('Creating input for x{}, with shape: {}: {}'.format(index, data.shape, data))

        input_layers.append(Input(shape=(data.shape[1],), name='x{}_input'.format(index)))

    layers = Concatenate()(input_layers)
    layers = Dense(32)(layers)
    layers = Dense(3)(layers)

    model = Model(input_layers, layers)

    model.compile(loss=losses.categorical_crossentropy, optimizer='adam')

    model.fit(input_layers, one_hot_labels)

    # TODO Register input placholders

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