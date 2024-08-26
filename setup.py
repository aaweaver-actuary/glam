from setuptools import setup, find_packages

setup(
name='glam',
version='0.1.0',  # Version number
    packages=find_packages(include=['src', 'src.glam.*'])
)

