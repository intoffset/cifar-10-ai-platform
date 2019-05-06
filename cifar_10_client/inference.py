#!/usr/bin/env python

import os
import numpy as np
import sys
from oauth2client.client import GoogleCredentials
from googleapiclient import discovery
from googleapiclient import errors

from absl import app, flags
import matplotlib.pyplot as plt

flags.DEFINE_string('project', None, "Project ID of GCP")
flags.DEFINE_string('model', None, "Model name")
flags.DEFINE_string('version', None, "Model version")
flags.DEFINE_bool('train_data', False, "Whether to use train data")
flags.DEFINE_string('input', '../input', "Input directory")
flags.DEFINE_integer('num', 1, "Number of image to inference")

FLAGS = flags.FLAGS

label_bytes = 1
image_bytes = 32 * 32 * 3


def main(argv=None):

    ml_service = discovery.build('ml', 'v1')

    label_names = np.loadtxt(os.path.join(FLAGS.input, "batches.meta.txt"), dtype=str)

    if FLAGS.train_data:
        filename = 'data_batch_1.bin'
    else:
        filename = 'test_batch.bin'

    with open(os.path.join(FLAGS.input, filename), 'rb') as f:
        for i in range(FLAGS.num):
            buffer = f.read(label_bytes + image_bytes)
            label = np.frombuffer(buffer, np.uint8, count=1, offset=0)[0]
            image = np.frombuffer(buffer, np.uint8, count=image_bytes, offset=1)
            image = np.transpose(np.reshape(image, [3, 32, 32]), [1, 2, 0])
            payload = generate_payload(image)
            res = request_inference(ml_service, payload, FLAGS.project, FLAGS.model, FLAGS.version)
            probabilities = res['predictions'][0]['probabilities']
            pred = res['predictions'][0]['classes']
            plt.title("Ground Truth: {}, Prediction: {} ({:.2f}%)".format(
                label_names[label], label_names[pred], probabilities[pred] * 100))
            plt.imshow(image)
            plt.show()


def generate_payload(image):
    image = image[2:30, 2:30, :]
    return {"instances": [{"x": image.tolist()}]}


def request_inference(service, payload, project, model, version=None):
    url = 'projects/{}/models/{}'.format(project, model)
    if version is not None:
        url += '/versions/{}'.format(version)

    try:
        request = service.projects().predict(
            name=url,
            body=payload
        )
        response = request.execute()
        return response

    except errors.HttpError as err:
        # Something went wrong, print out some information.
        print('There was an error creating the model. Check the details:')
        print(err._get_reason())


if __name__ == '__main__':
    app.run(main)
