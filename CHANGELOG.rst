Unreleased_
-----------

Added
^^^^^

* `@senwu`_: Support python 3.10 (`#123 <https://github.com/senwu/emmental/pull/123>`_)


0.1.1_ - 2022-01-11
-------------------

Fixed
^^^^^

* `@lorr1`_: Fix multiple wand issues.
  (`#118 <https://github.com/senwu/emmental/pull/118>`_,
  `#119 <https://github.com/senwu/emmental/pull/119>`_)
* `@senwu`_: Fix scikit-learn version.
  (`#120 <https://github.com/senwu/emmental/pull/120>`_)

0.1.0_ - 2021-11-24
-------------------

Deprecated
^^^^^^^^^^

* `@senwu`_: Deprecated argument `active` in learner and loss function api, and
  deprecated `ignore_index` argument in configuration.
  (`#107 <https://github.com/senwu/emmental/pull/107>`_)

Fixed
^^^^^

* `@senwu`_: Fix the metric cannot calculate issue when scorer is none.
  (`#112 <https://github.com/senwu/emmental/pull/112>`_)
* `@senwu`_: Fix Meta.config is None issue in collate_fn with num_workers > 1 when
  using python 3.8+ on mac.
  (`#117 <https://github.com/senwu/emmental/pull/117>`_)


Added
^^^^^

* `@senwu`_: Introduce two new classes: `Action` and `Batch` to make the APIs more
  modularized and make Emmental more extendable and easy to use for downstream tasks.
  (`#116 <https://github.com/senwu/emmental/pull/116>`_)

.. note::

    1. We introduce two new classes: `Action` and `Batch` to make the APIs more
    modularized.

    - `Action` are objects that populate the `task_flow` sequence. It has three
      attributes: name, module and inputs where name is the name of the action, module
      is the module name of the action and inputs is the inputs to the action. By
      introducing a class for specifying actions in the `task_flow`, we standardize its
      definition. Moreover,  `Action` enables more user flexibility in specifying a
      task flow as we can now support a wider-range of formats for the input attribute
      of a `task_flow` as discussed in (2).

    - `Batch` is the object that is returned from the Emmental `Scheduler`. Each
      `Batch` object has 6 attributes: uids (uids of the samples), X_dict (input
      features of the samples), Y_dict (output of the samples), task_to_label_dict
      (the task to label mapping), data_name (name of the dataset that samples come
      from), and split (the split information). By defining the `Batch` class, we unify
      and standardize the training scheduler interface by ensuring a consistent output
      format for all schedulers.

    2. We make the `task_flow` more flexible by supporting more formats for specifying
    inputs to each module.

    - It now supports str as inputs (e.g., inputs="input1") which means take the
      `input1`'s output as input for current action.

    - It also supports a list as inputs which can be constructed by three
      different formats:

      - x (x is str) where takes whole output of x's output as input: this enables
        users to pass all outputs from one module to another without having to
        manually specify every input to the module.

      - (x, y) (y is int) where takes x's y-th output as input.

      - (x, y) (y is str) where takes x's output str as input.

    Few emmental.Action examples:

    .. code:: python

      from emmental.Action as Act
      Act(name="input", module="input_module0", inputs=[("_input_", "data")])
      Act(name="input", module="input_module0", inputs=[("_input_", 0)])
      Act(name="input", module="input_module0", inputs=["_input_"])
      Act(name="input", module="input_module0", inputs="_input_")
      Act(name="input", module="input_module0", inputs=[("_input_", "data"), ("_input_", 1), "_input_"])
      Act(name="input", module="input_module0", inputs=None)

    This design also can be applied to action_outputs, here are few example:

    .. code:: python

      action_outputs=[(f"{task_name}_pred_head", 0), ("_input_", "data"), f"{task_name}_pred_head"]
      action_outputs="_input_"


0.0.9_ - 2021-10-05
-------------------

Added
^^^^^

* `@senwu`_: Support wandb logging.
  (`#99 <https://github.com/senwu/emmental/pull/99>`_)
* `@senwu`_: Fix log writer cannot dump functions in Meta.config issue.
  (`#103 <https://github.com/senwu/emmental/pull/103>`_)
* `@senwu`_: Add `return_loss` argument model predict and forward to support the case
  when no loss calculation can be done or needed.
  (`#105 <https://github.com/senwu/emmental/pull/105>`_)
* `@lorr1`_ and `@senwu`_: Add `skip_learned_data` to support skip trained data in
  learning.
  (`#101 <https://github.com/senwu/emmental/pull/101>`_,
  `#108 <https://github.com/senwu/emmental/pull/108>`_)

Fixed
^^^^^

* `@senwu`_: Fix model learning that cannot handle task doesn't have Y_dict from
  dataloasder such as contrastive learning.
  (`#105 <https://github.com/senwu/emmental/pull/105>`_)

0.0.8_ - 2021-02-14
-------------------

Added
^^^^^

* `@senwu`_: Support fp16 optimization.
  (`#77 <https://github.com/SenWu/emmental/pull/77>`_)
* `@senwu`_: Support distributed learning.
  (`#78 <https://github.com/SenWu/emmental/pull/78>`_)
* `@senwu`_: Support no label dataset.
  (`#79 <https://github.com/SenWu/emmental/pull/79>`_)
* `@senwu`_: Support output model immediate_ouput.
  (`#80 <https://github.com/SenWu/emmental/pull/80>`_)

.. note::

    To output model immediate_ouput, the user needs to specify which module output
    he/she wants to output in `EmmentalTask`'s `action_outputs`. It should be a pair of
    task_flow name and index or list of that pair. During the prediction phrase, the
    user needs to set `return_action_outputs=True` to get the outputs where the key is
    `{task_flow name}_{index}`.

    .. code:: python

        task_name = "Task1"
        EmmentalTask(
            name=task_name,
            module_pool=nn.ModuleDict(
                {
                    "input_module": nn.Linear(2, 8),
                    f"{task_name}_pred_head": nn.Linear(8, 2),
                }
            ),
            task_flow=[
                {
                    "name": "input",
                    "module": "input_module",
                    "inputs": [("_input_", "data")],
                },
                {
                    "name": f"{task_name}_pred_head",
                    "module": f"{task_name}_pred_head",
                    "inputs": [("input", 0)],
                },
            ],
            loss_func=partial(ce_loss, task_name),
            output_func=partial(output, task_name),
            action_outputs=[
                (f"{task_name}_pred_head", 0),
                ("_input_", "data"),
                (f"{task_name}_pred_head", 0),
            ],
            scorer=Scorer(metrics=task_metrics[task_name]),
        )

* `@senwu`_: Support action output dict.
  (`#82 <https://github.com/SenWu/emmental/pull/82>`_)
* `@senwu`_: Add a new argument `online_eval`. If `online_eval` is off, then model won't
  return `probs`.
  (`#89 <https://github.com/SenWu/emmental/pull/89>`_)
* `@senwu`_: Support multiple device training and inference.
  (`#91 <https://github.com/SenWu/emmental/pull/91>`_)

.. note::

    To train model on multiple devices such as CPU and GPU, the user needs to specify
    which module is on which device in `EmmentalTask`'s `module_device`. It's a
    ditctionary with key as the module_name and value as device number. During the
    training and inference phrase, the `Emmental` will automatically perform forward
    pass based on module device information.

    .. code:: python

        task_name = "Task1"
        EmmentalTask(
            name=task_name,
            module_pool=nn.ModuleDict(
                {
                    "input_module": nn.Linear(2, 8),
                    f"{task_name}_pred_head": nn.Linear(8, 2),
                }
            ),
            task_flow=[
                {
                    "name": "input",
                    "module": "input_module",
                    "inputs": [("_input_", "data")],
                },
                {
                    "name": f"{task_name}_pred_head",
                    "module": f"{task_name}_pred_head",
                    "inputs": [("input", 0)],
                },
            ],
            loss_func=partial(ce_loss, task_name),
            output_func=partial(output, task_name),
            action_outputs=[
                (f"{task_name}_pred_head", 0),
                ("_input_", "data"),
                (f"{task_name}_pred_head", 0),
            ],
            module_device={"input_module": -1, f"{task_name}_pred_head": 0},
            scorer=Scorer(metrics=task_metrics[task_name]),
        )

* `@senwu`_: Add require_prob_for_eval and require_pred_for_eval to optimize score
  function performance.
  (`#92 <https://github.com/SenWu/emmental/pull/92>`_)

.. note::

    The current approach during score the model will store probs and preds which might
    require a lot of memory resources especially for large datasets. The score function
    is also used in training. To optimize the score function performance, this PR
    introduces two new arguments in `EmmentalTask`: `require_prob_for_eval` and
    `require_pred_for_eval` which automatically selects whether `return_probs` or
    `return_preds`.

    .. code:: python

        task_name = "Task1"
        EmmentalTask(
            name=task_name,
            module_pool=nn.ModuleDict(
                {
                    "input_module": nn.Linear(2, 8),
                    f"{task_name}_pred_head": nn.Linear(8, 2),
                }
            ),
            task_flow=[
                {
                    "name": "input",
                    "module": "input_module",
                    "inputs": [("_input_", "data")],
                },
                {
                    "name": f"{task_name}_pred_head",
                    "module": f"{task_name}_pred_head",
                    "inputs": [("input", 0)],
                },
            ],
            loss_func=partial(ce_loss, task_name),
            output_func=partial(output, task_name),
            action_outputs=[
                (f"{task_name}_pred_head", 0),
                ("_input_", "data"),
                (f"{task_name}_pred_head", 0),
            ],
            module_device={"input_module": -1, f"{task_name}_pred_head": 0},
            require_prob_for_eval=True,
            require_pred_for_eval=True,
            scorer=Scorer(metrics=task_metrics[task_name]),
        )

* `@senwu`_: Support save and load optimizer and lr_scheduler checkpoints.
  (`#93 <https://github.com/SenWu/emmental/pull/93>`_)
* `@senwu`_: Support step based learning and add argument `start_step` and `n_steps` to
  set starting step and total step size.
  (`#93 <https://github.com/SenWu/emmental/pull/93>`_)


Fixed
^^^^^

* `@senwu`_: Fix customized optimizer support issue.
  (`#81 <https://github.com/SenWu/emmental/pull/81>`_)
* `@senwu`_: Fix loss logging didn't count task weight.
  (`#93 <https://github.com/SenWu/emmental/pull/93>`_)


0.0.7_ - 2020-06-03
-------------------

Added
^^^^^

* `@senwu`_: Support gradient accumulation step when machine cannot run large batch size.
  (`#74 <https://github.com/SenWu/emmental/pull/74>`_)
* `@senwu`_: Support user specified parameter groups in optimizer.
  (`#74 <https://github.com/SenWu/emmental/pull/74>`_)

.. note::

    When building the emmental learner, user can specify parameter groups for optimizer
    using `emmental.Meta.config["learner_config"]["optimizer_config"]["parameters"]`
    which is function takes the model as input and outputs a list of parameter groups,
    otherwise learner will create a parameter group with all parameters in the model.
    Below is an example of optimizing Adam Bert.

    .. code:: python

        def grouped_parameters(model):
            no_decay = ["bias", "LayerNorm.weight"]
            return [
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": emmental.Meta.config["learner_config"][
                        "optimizer_config"
                    ]["l2"],
                },
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]

        emmental.Meta.config["learner_config"]["optimizer_config"][
            "parameters"
        ] = grouped_parameters

Changed
^^^^^^^

* `@senwu`_: Enabled "Type hints (PEP 484) support for the Sphinx autodoc extension."
  (`#69 <https://github.com/SenWu/emmental/pull/69>`_)
* `@senwu`_: Refactor docstrings and enforce using flake8-docstrings.
  (`#69 <https://github.com/SenWu/emmental/pull/69>`_)

0.0.6_ - 2020-04-07
-------------------

Added
^^^^^

* `@senwu`_: Support probabilistic gold label in scorer.
* `@senwu`_: Add `add_tasks` to support adding one task or mulitple tasks into model.
* `@senwu`_: Add `use_exact_log_path` to support using exact log path.

.. note::

    When init the emmental there is one extra argument `use_exact_log_path` to use
    exact log path.

    .. code:: python

        emmental.init(dirpath, use_exact_log_path=True)

Changed
^^^^^^^

* `@senwu`_: Change running evaluation only when evaluation is triggered.


0.0.5_ - 2020-03-01
-------------------

Added
^^^^^

* `@senwu`_: Add `checkpoint_all` to controll whether to save all checkpoints.
* `@senwu`_: Support `CosineAnnealingLR`, `CyclicLR`, `OneCycleLR`, `ReduceLROnPlateau`
  lr scheduler.
* `@senwu`_: Support more unit tests.
* `@senwu`_: Support all pytorch optimizers.
* `@senwu`_: Support accuracy@k metric.
* `@senwu`_: Support cosine annealing lr scheduler.

Fixed
^^^^^

* `@senwu`_: Fix multiple checkpoint_metric issue.

0.0.4_ - 2019-11-11
-------------------

Added
^^^^^

* `@senwu`_: Log metric dict into log file every trigger evaluation time or full epoch.
* `@senwu`_: Add `get_num_batches` to calculate the total number batches from all
  dataloaders.
* `@senwu`_: Add `n_batches` in `EmmentalDataLoader` and `fillup` in `Scheduler` to
  support customize dataloader.
* `@senwu`_: Add overall and task specific loss during evaluating as default.
  to support user needs for clear checkpoins.
* `@senwu`_: Add `min_len` and `max_len` in `Meta.config` to support setting sequence
  length.
* `@senwu`_: Add overall and task specific loss during evaluating as default.
* `@senwu`_: Calculate overall and task specific metrics and loss during training.
* `@senwu`_: Add more util functions, e.g., array_to_numpy, construct_identifier,
  and random_string.
* `@senwu`_: Enforce dataset has uids attribute.
* `@senwu`_: Add micro/macro metric options which have split-wise micro/macro average
  and global-wise micro/macro average. The name for the metrics are:

::

  split-wise micro average: `model/all/{split}/micro_average`
  split-wise macro average: `model/all/{split}/macro_average`
  global-wise micro average: `model/all/all/micro_average`
  global-wise macro average: `model/all/all/macro_average`

*Note*: `micro` means average all metrics from all tasks. `macro` means average all
  average metric from all tasks.

* `@senwu`_: Add contrib folder to support unofficial usages.

Fixed
^^^^^

* `@senwu`_: Correct lr update for epoch-wised scheduler.
* `@senwu`_: Add type for class.
* `@senwu`_: Add warning for one class in ROC AUC metric.
* `@senwu`_: Fix missing support for StepLR and MultiStepLR lr scheduler.
* `@senwu`_: Fix missing pytest.ini and fix test cannot remove temp dir issue.
* `@senwu`_: Fix default train loss metric from `model/train/all/loss` to
  `model/all/train/loss` to follow the format `TASK_NAME/DATA_NAME/SPLIT/METRIC`
  pattern.

Changed
^^^^^^^

* `@senwu`_: Change default grad clip to None.
* `@senwu`_: Update seed and grad_clip to nullable.
* `@senwu`_: Change default class index to 0-index.
* `@senwu`_: Change default ignore_index to None.
* `@senwu`_: Change the default counter unit to epoch.
* `@senwu`_: Update the metric to return one metric value by default.

Removed
^^^^^^^

* `@senwu`_: Remove `checkpoint_clear` argument.

.. _Unreleased: https://github.com/senwu/emmental/compare/v0.1.1...main
.. _0.0.4: https://github.com/senwu/emmental/compare/v0.0.3...v0.0.4
.. _0.0.5: https://github.com/senwu/emmental/compare/v0.0.4...v0.0.5
.. _0.0.6: https://github.com/senwu/emmental/compare/v0.0.5...v0.0.6
.. _0.0.7: https://github.com/senwu/emmental/compare/v0.0.6...v0.0.7
.. _0.0.8: https://github.com/senwu/emmental/compare/v0.0.7...v0.0.8
.. _0.0.9: https://github.com/senwu/emmental/compare/v0.0.8...v0.0.9
.. _0.1.0: https://github.com/senwu/emmental/compare/v0.0.9...v0.1.0
.. _0.1.1: https://github.com/senwu/emmental/compare/v0.1.0...v0.1.1
..
  For convenience, all username links for contributors can be listed here

.. _@senwu: https://github.com/senwu
.. _@lorr1: https://github.com/lorr1
