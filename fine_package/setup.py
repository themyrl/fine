from setuptools import setup, find_namespace_packages

setup(name='fine',
      packages=find_namespace_packages(include=["fine", "fine.*"]),
      version='0.0.1',
      install_requires=[
            "tensorboard",
            "einops",
            "timm",
            "matplotlib",
            ]
      )
