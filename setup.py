'''
This file is an essential part of packaging and
distributing Python projects. It is used by setuptools
(or distutils in older Python versions) to define the configuration
of your projects, such as its metadata, dependencies, etc.
'''

from setuptools import find_packages, setup
from typing import List


def get_requirements() -> List[str]:
    """
    Thiss function will return list of requirements

    """
    requirement_lst: List[str] = []
    try:
        with open('project_requirements.txt', 'r') as file:
            # Read lines from the file
            lines = file.readlines()
            ## Process each line
            for line in lines:
                requirement = line.strip()
                ## ignore empty lines and -e .
                if requirement and requirement not in ('-e .', '-e'):
                    requirement_lst.append(requirement)
    except FileNotFoundError:
        print("requirements.txt file not found")

    return requirement_lst


setup(
    name="NetworkSecurity",
    version="0.0.1",
    author="Ziad Waleed",
    author_email="ziadwaleed9@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements()
)