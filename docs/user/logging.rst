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
        counter_unit: epoch # [epoch, batch]
        evaluation_freq: 1
        writer_config:
            writer: tensorboard # [json, tensorboard, wandb]
            verbose: True
            wandb_project_name:
            wandb_run_name:
            wandb_watch_model: False
            wandb_model_watch_freq:
            write_loss_per_step: False
        checkpointing: False
        checkpointer_config:
            checkpoint_path:
            checkpoint_freq: 1
            checkpoint_metric:
                model/train/all/loss: min # metric_name: mode, where mode in [min, max]
            checkpoint_task_metrics: # task_metric_name: mode
            checkpoint_runway: 0 # checkpointing runway (no checkpointing before k unit)
            checkpoint_all: False # checkpointing all checkpoints
            clear_intermediate_checkpoints: True # whether to clear intermediate checkpoints
            clear_all_checkpoints: False # whether to clear all checkpoints


.. _Configuring Emmental: config.html
.. _Emmental: https://github.com/SenWu/Emmental
