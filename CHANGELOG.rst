[Unreleased]
------------

Added
^^^^^
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
* `@senwu`_: Change the default counter unit to epoch.
* `@senwu`_: Update the metric to return one metric value by default.

Removed
^^^^^^^
* `@senwu`_: Remove `checkpoint_clear` argument.

..
  For convenience, all username links for contributors can be listed here

.. _@senwu: https://github.com/senwu
