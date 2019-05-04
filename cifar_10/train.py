#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
from absl import app, flags
import tensorflow as tf
from datetime import datetime

from cifar_10.input import get_train_dataset, get_test_dataset
from cifar_10.model import gen_model

flags.DEFINE_string('input', '../input', "input directory")
flags.DEFINE_string('output', '../output', "output directory")
flags.DEFINE_integer('batch', 64, "batch size")
flags.DEFINE_integer('epochs', 10, "epochs")

FLAGS = flags.FLAGS

num_train_data = 50000
num_test_data = 10000


def main(argv=None):

    # DEFINE INPUT
    train_dataset = get_train_dataset(FLAGS.input, FLAGS.batch)
    test_dataset = get_test_dataset(FLAGS.input, FLAGS.batch)

    # DEFINE MODEL
    model = gen_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    dir_log = os.path.join(FLAGS.output, 'log')
    dir_model = os.path.join(FLAGS.output, 'cifar-10-model', str(int(datetime.now().timestamp())))

    os.makedirs(dir_log, exist_ok=True)

    # Save configuration
    path_flags = os.path.join(dir_log, 'flags.json')
    with open(path_flags, 'w', encoding='utf-8') as f:
        json.dump(FLAGS.flag_values_dict(), f, ensure_ascii=False, indent=4)

    # Save model summary
    path_model_summary = os.path.join(dir_log, 'model.txt')
    with open(path_model_summary, 'w') as f:
        model.summary(print_fn=lambda x: print(x, file=f))

    steps_pre_epoch = num_train_data / FLAGS.batch
    model.fit(train_dataset, epochs=FLAGS.epochs, steps_per_epoch=steps_pre_epoch,
              validation_data=test_dataset)

    tf.saved_model.save(model, dir_model)


if __name__ == '__main__':
    app.run(main)