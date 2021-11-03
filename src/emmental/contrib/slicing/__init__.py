"""Emmental slicing contrib module."""
from emmental.contrib.slicing.data import add_slice_labels
from emmental.contrib.slicing.slicing_function import SlicingFunction
from emmental.contrib.slicing.task import build_slice_tasks

__all__ = ["add_slice_labels", "SlicingFunction", "build_slice_tasks"]
