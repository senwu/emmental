Getting Started
===============

Should see something here:

.. automodule:: emmental.metrics
    :members:
    :inherited-members:
    :show-inheritance:

Before this line.


This document will show you how to get up and running with Emmental. We'll show
you how to get everything installed and your machine so that you can walk
through real examples by checking out our Tutorials_.

Installing the Emmental Package
-------------------------------

Then, install Emmental by running::

    $ pip install emmental

.. note::
    Emmental only supports Python 3. Python 2 is not supported.

.. tip::
  For the Python dependencies, we recommend using a virtualenv_, which will
  allow you to install Emmental and its python dependencies in an isolated
  Python environment. Once you have virtualenv installed, you can create a
  Python 3 virtual environment as follows.::

      $ virtualenv -p python3.6 .venv

  Once the virtual environment is created, activate it by running::

      $ source .venv/bin/activate

  Any Python libraries installed will now be contained within this virtual
  environment. To deactivate the environment, simply run::

      $ deactivate


The Emmental Framework
----------------------

The Emmental framework can be broken into four components.

  #. Dataset
  #. Task
  #. Model
  #. Learning

To demonstrate how to set up and use Emmental in your applications, we walk
through each of these phases in real-world examples in our Tutorials_.

.. _Tutorials: https://github.com/SenWu/emmental-tutorials
.. _homebrew: https://brew.sh
.. _virtualenv: https://virtualenv.pypa.io/en/stable/
