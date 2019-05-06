#!/usr/bin/env python

from oauth2client.client import GoogleCredentials
from googleapiclient import discovery
from googleapiclient import errors

from absl import app, flags

flags.DEFINE_string('project', None, "Project ID of GCP")
flags.DEFINE_string('model', None, "Model name")
flags.DEFINE_string('version', None, "Model version")

FLAGS = flags.FLAGS


def main(argv=None):
    ml_service = discovery.build('ml', 'v1')
    res = get_model_meta(ml_service, FLAGS.project, FLAGS.model, FLAGS.version)
    print(res)


def get_model_meta(service, project, model, version=None):
    url = 'projects/{}/models/{}'.format(project, model)
    if version:
        url += '/versions/{}'.format(version)
        response = service.projects().models().versions().get(name=url).execute()
        return response
    else:
        response = service.projects().models().get(name=url).execute()
        return response['defaultVersion']


if __name__ == '__main__':
    app.run(main)
