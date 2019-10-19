Model
=====
The third component of Emmental_'s pipeline is to build deep learning model with
your tasks.

Emmental_ Model
---------------
The following describes elements of used for model creation.

.. automodule:: emmental.model
    :members:
    :show-inheritance:

Configuration Settings
----------------------

Visit the `Configuring Emmental`_ page to see how to provide configuration
parameters to Emmental_ via ``.emmental-config.yaml``.

The model parameters are described below::

    # Model configuration
    model_config:
        model_path: # path to pretrained model
        device: 0 # -1 for cpu or gpu id (e.g., 0 for cuda:0)
        dataparallel: True # whether to use dataparallel or not


.. _Configuring Emmental: config.html
.. _Emmental: https://github.com/SenWu/Emmental
