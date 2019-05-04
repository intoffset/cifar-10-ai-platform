from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['tensorflow-gpu==2.0.0a0']

setup(
    name='cifar_10',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='AI Platform example for CIFAR-10 model.')