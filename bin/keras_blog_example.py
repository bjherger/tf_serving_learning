# Courtesy https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html#exporting-a-model-with-tensorflow-serving

from keras import backend as K, Sequential
import tensorflow as tf
from keras.engine import InputLayer
from keras.layers import Dense
from tensorflow.contrib.session_bundle import exporter

sess = tf.Session()

img = tf.placeholder(tf.float32, shape=(None, 784))
custom_input_tensor = img

K.set_session(sess)

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=784))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# this works!
# x = tf.placeholder(tf.float32, shape=(None, 784))
# y = model(x)

# from session_bundle import exporter


export_path = '~/Downloads'  # where to save the exported graph
export_version = 2  # version number (integer)

sess.run(tf.global_variables_initializer())

saver = tf.train.Saver(sharded=False)
model_exporter = exporter.Exporter(saver)
signature = exporter.classification_signature(input_tensor=model.input,
                                              scores_tensor=model.output)
model_exporter.init(sess.graph.as_graph_def(),
                    default_graph_signature=signature)
model_exporter.export(export_path, tf.constant(export_version), sess)
