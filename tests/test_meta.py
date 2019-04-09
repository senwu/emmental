#! /usr/bin/env python
import logging
import os
import shutil

import emmental


def test_meta(caplog):
    """Unit test of meta"""

    caplog.set_level(logging.INFO)

    dirpath = "temp_test_meta_log_folder"

    emmental.init_logging(dirpath)

    # Check the log folder is created correctly
    assert os.path.isdir(dirpath) is True

    shutil.rmtree(dirpath)
