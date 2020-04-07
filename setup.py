"""For pip."""
from glob import glob
from os.path import basename, splitext

from setuptools import find_packages, setup

exec(open("src/emmental/_version.py").read())
setup(
    name="emmental",
    version=__version__,
    description="A framework for building multi-modal multi-task learning systems.",
    long_description=open("README.rst").read(),
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={"": ["emmental-default-config.yaml"]},
    py_modules=[splitext(basename(path))[0] for path in glob("src/*.py")],
    install_requires=[
        "numpy>=1.11, <2.0",
        "pyyaml>=5.1, <6.0",
        "scikit-learn>=0.20.0, <0.30.0",
        "scipy>=1.1.0, <2.0.0",
        "tensorboard>=1.15.0, <3.0.0",
        "torch>=1.3.1, <2.0.0",
        "tqdm>=4.36.0, <5.0.0",
    ],
    keywords=["emmental", "multi task learning", "deep learing"],
    include_package_data=True,
    url="https://github.com/SenWu/emmental",
    classifiers=[  # https://pypi.python.org/pypi?:action=list_classifiers
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3 :: Only",
    ],
    project_urls={
        "Tracker": "https://github.com/SenWu/emmental/issues",
        "Source": "https://github.com/SenWu/emmental",
    },
    python_requires=">=3.6",
    author="Sen Wu",
    author_email="senwu@cs.stanford.edu",
    license="MIT",
)
