Logging
=======

This page shows descriptions of the logging functions included with Emmental which
logs the learning information and checkpoints.

Logging Classes
------------------------------------

The following docs describe elements of Emmental_'s logging utilites.

.. automodule:: emmental.logging
    :members:
    :inherited-members:
    :show-inheritance:

Configuration Settings
----------------------

Visit the `Configuring Emmental`_ page to see how to provide configuration
parameters to Emmental_ via ``.emmental-config.yaml``.

The logging parameters of Emmental are described below::

    # Logging configuration
    logging_config:
        counter_unit: batch # [epoch, batch]
        evaluation_freq: 2
        writer_config:
            writer: tensorboard # [json, tensorboard]
            verbose: True
        checkpointing: True
        checkpointer_config:
            checkpoint_path:
            checkpoint_freq: 1
            checkpoint_metric: # metric_name: mode, where mode in [min, max]
                # model/train/all/loss: min
            checkpoint_task_metrics: # task_metric_name: mode
            checkpoint_runway: 0 # checkpointing runway (no checkpointing before k unit)
            checkpoint_clear: True # whether to clear immedidate checkpointing

.. _Configuring Emmental: config.html
.. _Emmental: https://github.com/SenWu/Emmental
