from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT='-e .'

def get_requirements(file_path)-> List[str]:
    """Returns a list of requirements
    """
    requirements=[]
    with open(file_path) as f:
        requirements=f.read().splitlines()

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements


setup(
    name='mlproject',
    version='0.0.1',
    author='Samy',
    author_email='samy.nehlil@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    description='End to End Machine Learning Project'
)