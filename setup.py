"""For pip."""
from glob import glob
from os.path import basename, splitext

from setuptools import find_packages, setup

exec(open("src/multtm/_version.py").read())
setup(
    name="multtm",
    version=__version__,
    description="A generic deep learning framework for multi-task learning.",
    long_description=open("README.rst").read(),
    packages=find_packages("src"),
    package_dir={"": "src"},
    py_modules=[splitext(basename(path))[0] for path in glob("src/*.py")],
    install_requires=[
        "numpy>=1.11, <2.0",
        "pandas>=0.23.4, <0.24.0",
        "pyyaml>=4.2b1, <5.0",
        "scipy>=1.1.0, <2.0.0",
        "tensorboardX>=1.6, <2.0",
        "torch>=1.0, <2.0",
        "tqdm>=4.26.0, <5.0.0",
    ],
    keywords=["multim", "multi task learning", "deep learing"],
    include_package_data=True,
    url="https://github.com/SenWu/MultTM",
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
        "Tracker": "https://github.com/SenWu/MultTM/issues",
        "Source": "https://github.com/SenWu/MultTM",
    },
    python_requires=">=3.6",
    author="Sen Wu",
    author_email="senwu@cs.stanford.edu",
    license="MIT",
)
