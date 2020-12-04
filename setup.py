'''Setup script
Usage: pip install .
To install development dependencies too, run: pip install .[dev]
'''
from setuptools import setup, find_packages

setup(
    name='stackgan',
    version='v1',
    packages=find_packages(),
    scripts=[],
    # url = ,
    author='Georgios Chochlakis',
    install_requires=[],
    extras_require={
        'dev': [
            'torch',
            'torchvision',
            'pandas',
            'sklearn',
            'scipy'
        ],
    },
)
