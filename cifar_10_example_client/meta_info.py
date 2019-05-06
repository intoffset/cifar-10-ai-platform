#!/usr/bin/env python

from oauth2client.client import GoogleCredentials
from googleapiclient import discovery
from googleapiclient import errors

import click


@click.command()
@click.option('--project', required=True, help="GCP project ID")
@click.option('--model', required=True, help="model name")
@click.option('--version', default=None, required=False, help="model version")
def main(project, model, version):
    ml_service = discovery.build('ml', 'v1')
    res = get_model_meta(ml_service, project, model, version)
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
    main()
