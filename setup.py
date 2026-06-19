from setuptools import find_packages, setup
from typing import List


def get_requirements(file_path: str) -> List[str]:
    with open(file_path, "r") as file:
        requirements = file.read().splitlines()
    return requirements


setup(
    name="PRODIGY_ML_02",
    version="0.1.0",
    author="Pruthviraj Gade",
    author_email="pruthvirajgade2@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
