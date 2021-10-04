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

Schedulers
-------------------

These are several schedulers supported in Emmental_ learner.

.. automodule:: emmental.schedulers
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
        optimizer_path: # path to optimizer state
        scheduler_path: # path to lr scheduler state
        fp16: False # whether to use 16-bit precision
        fp16_opt_level: O1 # Apex AMP optimization level (e.g., ['O0', 'O1', 'O2', 'O3'])
        local_rank: -1 # local_rank for distributed training on gpus
        epochs_learned: 0 # learning epochs learned
        n_epochs: 1 # total number of learning epochs
        steps_learned: 0 # learning steps learned
        n_steps: # total number of learning steps
        skip_learned_data: False # skip learned batches if steps_learned or epochs_learned nonzero
        train_split: # the split for training, accepts str or list of strs
            - train
        valid_split: # the split for validation, accepts str or list of strs
            - valid
        test_split: # the split for testing, accepts str or list of strs
            - test
        ignore_index: # the ignore index, uses for masking samples
        online_eval: 0 # whether to perform online evaluation
        optimizer_config:
            optimizer: adam # [sgd, adam, adamax, bert_adam]
            parameters: # parameters to optimize
            lr: 0.001 # Learing rate
            l2: 0.0 # l2 regularization
            grad_clip: # gradient clipping
            gradient_accumulation_steps: 1 # gradient accumulation steps
            asgd_config:
                lambd: 0.0001
                alpha: 0.75
                t0: 1000000.0
            adadelta_config:
                rho: 0.9
                eps: 0.000001
            adagrad_config:
                lr_decay: 0
                initial_accumulator_value: 0
                eps: 0.0000000001
            adam_config:
                betas: !!python/tuple [0.9, 0.999]
                eps: 0.00000001
                amsgrad: False
            adamw_config:
                betas: !!python/tuple [0.9, 0.999]
                eps: 0.00000001
                amsgrad: False
            adamax_config:
                betas: !!python/tuple [0.9, 0.999]
                eps: 0.00000001
            lbfgs_config:
                max_iter: 20
                max_eval:
                tolerance_grad: 0.0000001
                tolerance_change: 0.000000001
                history_size: 100
                line_search_fn:
            rms_prop_config:
                alpha: 0.99
                eps: 0.00000001
                momentum: 0
                centered: False
            r_prop_config:
                etas: !!python/tuple [0.5, 1.2]
                step_sizes: !!python/tuple [0.000001, 50]
            sgd_config:
                momentum: 0
                dampening: 0
                nesterov: False
            sparse_adam_config:
                betas: !!python/tuple [0.9, 0.999]
                eps: 0.00000001
            bert_adam_config:
                betas: !!python/tuple [0.9, 0.999]
                eps: 0.00000001
        lr_scheduler_config:
            lr_scheduler: # [linear, exponential, reduce_on_plateau, cosine_annealing]
            lr_scheduler_step_unit: batch # [batch, epoch]
            lr_scheduler_step_freq: 1
            warmup_steps: # warm up steps
            warmup_unit: batch # [epoch, batch]
            warmup_percentage: # warm up percentage
            min_lr: 0.0 # minimum learning rate
            reset_state: False # reset the state of the optimizer
            exponential_config:
                gamma: 0.9
            plateau_config:
                metric: model/train/all/loss
                mode: min
                factor: 0.1
                patience: 10
                threshold: 0.0001
                threshold_mode: rel
                cooldown: 0
                eps: 0.00000001
            step_config:
                step_size: 1
                gamma: 0.1
                last_epoch: -1
            multi_step_config:
                milestones:
                    - 1000
                gamma: 0.1
                last_epoch: -1
            cyclic_config:
                base_lr: 0.001
                max_lr: 0.1
                step_size_up: 2000
                step_size_down:
                mode: triangular
                gamma: 1.0
                scale_fn:
                scale_mode: cycle
                cycle_momentum: True
                base_momentum: 0.8
                max_momentum: 0.9
                last_epoch: -1
            one_cycle_config:
                max_lr: 0.1
                pct_start: 0.3
                anneal_strategy: cos
                cycle_momentum: True
                base_momentum: 0.85
                max_momentum: 0.95
                div_factor: 25.0
                final_div_factor: 10000.0
                last_epoch: -1
            cosine_annealing_config:
                last_epoch: -1
        task_scheduler_config:
            task_scheduler: round_robin # [sequential, round_robin, mixed]
            sequential_scheduler_config:
                fillup: False
            round_robin_scheduler_config:
                fillup: False
            mixed_scheduler_config:
                fillup: False
        global_evaluation_metric_dict: # global evaluation metric dict


.. _Configuring Emmental: config.html
.. _Emmental: https://github.com/SenWu/Emmental
