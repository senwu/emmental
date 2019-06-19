Learning
========
The final component of Emmental_'s pipeline is to learn the user defined deep learning
model based user defined data.

Core Learning Objects
---------------------

These are Emmental_'s core objects used for learning.

.. automodule:: emmental.learner
    :members:
    :inherited-members:
    :show-inheritance:

Configuration Settings
----------------------

Visit the `Configuring Emmental`_ page to see how to provide configuration
parameters to Emmental_ via ``.emmental-config.yaml``.

The learning parameters of the model are described below::

    # Learning configuration
    learner_config:
        fp16: False # whether to use half precision
        n_epochs: 1 # total number of learning epochs
        train_split: train # the split for training, accepts str or list of strs
        valid_split: valid # the split for validation, accepts str or list of strs
        test_split: test # the split for testing, accepts str or list of strs
        ignore_index: 0 # the ignore index, uses for masking samples
        optimizer_config:
            optimizer: adam # [sgd, adam, adamax, bert_adam]
            lr: 0.001 # Learing rate
            l2: 0.0 # l2 regularization
            grad_clip: 1.0 # gradient clipping
            sgd_config:
                momentum: 0.9
            adam_config:
                betas: !!python/tuple [0.9, 0.999]
                eps: 0.00000001
                amsgrad: False
            adamax_config:
                betas: !!python/tuple [0.9, 0.999]
                eps: 0.00000001
            bert_adam_config:
                betas: !!python/tuple [0.9, 0.999]
                eps: 0.00000001
        lr_scheduler_config:
            lr_scheduler: # [linear, exponential, reduce_on_plateau]
            warmup_steps: # warm up steps
            warmup_unit: batch # [epoch, batch]
            warmup_percentage: # warm up percentage
            min_lr: 0.0 # minimum learning rate
            linear_config:
                min_lr: 0.0
            exponential_config:
                gamma: 0.9
            plateau_config:
                factor: 0.5
                patience: 10
                threshold: 0.0001
        task_scheduler: round_robin # [sequential, round_robin]
        global_evaluation_metric_dict: # global evaluation metric dict

.. _Configuring Emmental: config.html
.. _Emmental: https://github.com/SenWu/Emmental
