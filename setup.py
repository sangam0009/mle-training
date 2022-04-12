# from xml.etree.ElementTree import VERSION
from setuptools import setup, find_packages

VERSION = "0.0.1"
DESCRIPTION = "A basic hello package"

setup(
    name="HelloPKG",
    version="VERSION",
    description="DESCRIPTION",
    url="https://github.com/sangam0009/mle-training",
    author="Sangam Venkat",
    author_email="venkat.sangam@tigeranalytics.com",
    license="MIT",
    packages=find_packages(),
    keywords=["python"],
    zip_safe=False,
)
