from setuptools import setup

setup(
    name="gym_continual_rl",
    version="0.0.2",
    install_requires=[
        "gymnasium==0.29.1",
        "numpy==1.26.4",
        "pygame==2.5.2",
    ],
    packages=["gym_continual_rl"],
    python_requires=">=3.9",
)
