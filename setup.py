#!/usr/bin/python

from setuptools import setup

setup(
    name="firm_growth_reg",
    version="0.0.2",
    description="",
    author="Leo Liu",
    author_email="leo.liu@unsw.edu.au",
    install_requires=[
        "matplotlib",
        "linearmodels",
        "scipy",
        "pandas",
        "icecream",
        "loguru",
        "latex_table @ git+https://github.com/leoliu0/latex_table@master",
    ],
    packages=["firm_growth_reg"],
)
