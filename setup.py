from setuptools import setup

with open('requirements.txt', 'r') as f:
    requirements = f.readlines()

setup(
    name='ca_prep',
    version='0.0.1',
    packages=['ca_prep', 'ca_prep.preprocessing'],
    url='',
    license='GPL v3',
    author='thladnik',
    author_email='tim.hladnik@gmail.com',
    description='',
    requirements=requirements
)
