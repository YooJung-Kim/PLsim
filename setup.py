from setuptools import setup, find_packages

setup(
    name="PLsim",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "hcipy",
        "lightbeam",
        "astropy"
    ],
)