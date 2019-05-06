import tensorflow as tf
from tensorflow.python.keras import models, layers
from tensorflow.python.keras.applications import MobileNetV2, DenseNet121


def gen_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model


def wrap_model(basemodel):
    class WrapperModel(tf.keras.Model):
        def __init__(self, base_model):
            super(WrapperModel, self).__init__()
            self.base_model = base_model

        def call(self, x):
            x = tf.cast(x, tf.float32)
            x = x / 255.
            probability = tf.identity(self.base_model(x), name='probabilities')
            classes = tf.argmax(probability, axis=1, name='classes')
            return classes, probability

    wrapper_model = WrapperModel(basemodel)

    if not wrapper_model.inputs:
        wrapper_model._set_inputs(tf.zeros(shape=(1, 28, 28, 3), dtype=tf.uint8), training=False)
    wrapper_model.input_names = ['x']
    wrapper_model.output_names = ['classes', 'probabilities']
    return wrapper_model


