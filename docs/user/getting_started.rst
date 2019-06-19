Getting Started
===============

This document will show you how to get up and running with Emmental. We'll show
you how to get everything installed on your machine so that you can walk
through real examples by checking out our Tutorials_.

Installing the Emmental Package
-------------------------------

Install Emmental by running::

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

  #. Dataset and Dataloader

      In this first component, the users' input is parsed into Emmental's dataset and
      then feed into Emmental's dataloader.

  #. Task

      In this component, we let user to use declarative language-like way to defeine
      the taksk, which includes task name (name), module used in the task (module_pool),
      task flow (task_flow), loss function used in the task (loss_func), output 
      function (output_func), and the score functions (scorer).

  #. Model

      Here, we initialize the Emmental model with the Emmental tasks. Users can define
      different types of models, such as single-task model, multi-task model,
      multi-modality task.

  #. Learning

      Finally, Emmental provides learning component which is used to train the Emmental
      model. Optionally, users can use different training schedulers during learning
      process.

To demonstrate how to set up and use Emmental in your applications, we walk
through each of these phases in real-world examples in our Tutorials_.

.. _Tutorials: https://github.com/SenWu/emmental-tutorials
.. _homebrew: https://brew.sh
.. _virtualenv: https://virtualenv.pypa.io/en/stable/
