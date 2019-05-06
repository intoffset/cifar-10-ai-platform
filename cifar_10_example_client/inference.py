#!/usr/bin/env python

import os

from oauth2client.client import GoogleCredentials
from googleapiclient import discovery
from googleapiclient import errors
import click
import matplotlib.pyplot as plt
import numpy as np


label_bytes = 1
image_bytes = 32 * 32 * 3


@click.command()
@click.option('--project', required=True, help="GCP project ID")
@click.option('--model', required=True, help="model name")
@click.option('--version', default=None, required=False, help="model version")
@click.option('--input', default='../input', required=False, help="input directory")
@click.option('--train-data/--no-train-data', default=False, required=False, help="whether to inference training data")
@click.option('--num', type=int, default=1, required=False, help="number of image to inference")
def main(project, model, version, train_data, input, num):
    ml_service = discovery.build('ml', 'v1')

    label_names = np.loadtxt(os.path.join(input, "batches.meta.txt"), dtype=str)

    if train_data:
        filename = 'data_batch_1.bin'
    else:
        filename = 'test_batch.bin'

    with open(os.path.join(input, filename), 'rb') as f:
        for i in range(num):
            buffer = f.read(label_bytes + image_bytes)
            label = np.frombuffer(buffer, np.uint8, count=1, offset=0)[0]
            image = np.frombuffer(buffer, np.uint8, count=image_bytes, offset=1)
            image = np.transpose(np.reshape(image, [3, 32, 32]), [1, 2, 0])
            payload = generate_payload(image)
            response = request_inference(ml_service, payload, project, model, version)
            probabilities = response['predictions'][0]['probabilities']
            pred = response['predictions'][0]['classes']
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
