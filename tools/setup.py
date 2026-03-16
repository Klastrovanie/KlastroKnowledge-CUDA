from setuptools import setup
from setuptools.extension import Extension

setup(
    name='klastroknowledge',
    version='0.0.0.1',
    packages=['klastroknowledge'],
    package_data={'klastroknowledge': ['*.so']},
)