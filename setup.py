from setuptools import setup, find_packages

setup(
    name='vetpack',
    version='0.0.dev1',
    author='Benjamin V. Rackham',
    author_email='brackham@mit.edu',
    description='Tools for vetting transit signals',
    long_description=open('README.md', 'r').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/disruptiveplanets/vetpack/',
    packages=find_packages(),
    install_requires=open('requirements.txt').read().split('\n'),
)
