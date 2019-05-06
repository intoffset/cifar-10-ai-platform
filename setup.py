from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'tensorflow-gpu==2.0.0a0',
    'oauth2client>=4.1.3',
    'google-api-python-client>=1.7.8',
    'matplotlib>=3.0.3'
]

setup(
    name='cifar_10_example',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='AI Platform example for CIFAR-10 model.')
