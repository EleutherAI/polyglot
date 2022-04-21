# Copyright 2021 TUNiB Inc.


from setuptools import find_packages, setup

VERSION = {}  # type: ignore

with open("multilingual/__version__.py", "r") as version_file:
    exec(version_file.read(), VERSION)

with open("requirements.txt", "r") as requirements_file:
    INSTALL_REQUIRES = requirements_file.read().splitlines()

setup(
    name="multilingual",
    version=VERSION["version"],
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/EleutherAI/multilingual-transfer",
    author="EleutherAI multilinaul transfer team",
    author_email="kevin.ko@tunib.ai",
    install_requires=INSTALL_REQUIRES,
    packages=find_packages(
        include=["multilingual", "multilingual.*"],
        exclude="tests",
    ),
    python_requires=">=3.6.0",
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    package_data={},
    dependency_links=[],
    include_package_data=True,
    zip_safe=False,
)
