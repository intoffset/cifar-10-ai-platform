from tensorflow.python.keras import models, layers
from tensorflow.python.keras.applications import MobileNetV2, DenseNet121


def gen_model():
    return MobileNetV2(input_shape=(32, 32, 3))
