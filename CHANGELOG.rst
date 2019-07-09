[Unreleased]
------------

Added
^^^^^
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
* `@senwu`_: Add missing pytest.ini and fix test cannot remove temp dir issue.
* `@senwu`_: Fix default train loss metric from `model/train/all/lr` to
  `model/all/train/lr` to follow the format `TASK_NAME/DATA_NAME/SPLIT/METRIC` pattern.

Changed
^^^^^^^
* `@senwu`_: Update the metric to return one metric value by default.

..
  For convenience, all username links for contributors can be listed here

.. _@senwu: https://github.com/senwu
