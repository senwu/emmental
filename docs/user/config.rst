Configuring Emmental
====================

By default, Emmental_ loads the default config ``.emmental-default-config.yaml``
from the Emmental_ directory, and loads the user defined config
``emmental-config.yaml`` starting from the current working directory, allowing you
to have multiple configuration files for different directories or projects. If it's
not there, it looks in parent directories. If no file is found, a default
configuration will be used.

Emmental will only ever use one ``.emmental-config.yaml`` file. It does not look
for multiple files and will not compose configuration settings from different
files.

The default ``.emmental-config.yaml`` configuration file is shown below::

    # Meta configuration
    meta_config:
        seed: # random seed for all numpy/torch/cuda operations in model and learning
        verbose: True # whether to print the log information
        log_path: logs # log directory
        use_exact_log_path: False # whether to use the exact log directory

    # Data configuration
    data_config:
        min_data_len: 0 # min data length
        max_data_len: 0 # max data length (e.g., 0 for no max_len)

    # Model configuration
    model_config:
        model_path: # path to pretrained model
        device: 0 # -1 for cpu or gpu id (e.g., 0 for cuda:0)
        dataparallel: True # whether to use dataparallel or not
        distributed_backend: nccl # what distributed backend to use for DDP [nccl, gloo]

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


User can also use the Emmental_ utility function ``parse_arg`` and
``parse_arg_to_config`` from ``emmental.utils`` to generate the config object.

.. _Emmental: https://github.com/SenWu/Emmental