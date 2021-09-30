Dataset and Dataloader
======================
The first component of Emmental_'s pipeline is to use user provided data to create
Emmental Dataset and Dataloader.

Dataset and Dataloader Classes
------------------------------------

The following docs describe elements of Emmental's Dataset and Dataloader.

.. automodule:: emmental.data
    :members:
    :inherited-members:
    :show-inheritance:

Configuration Settings
----------------------

Visit the `Configuring Emmental`_ page to see how to provide configuration
parameters to Emmental_ via ``.emmental-config.yaml``.

The parameters of data are described below::

    # Data configuration
    data_config:
        min_data_len: 0 # min data length
        max_data_len: 0 # max data length (e.g., 0 for no max_len)


.. _Configuring Emmental: config.html
.. _Emmental: https://github.com/SenWu/Emmental