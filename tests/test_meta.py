import logging
import os
import shutil

import emmental
from emmental import Meta


def test_meta(caplog):
    """Unit test of meta"""

    caplog.set_level(logging.INFO)

    dirpath = "temp_test_meta_log_folder"

    Meta.reset()
    emmental.init(dirpath)

    # Check the log folder is created correctly
    assert os.path.isdir(dirpath) is True
    assert Meta.log_path.startswith(dirpath) is True

    # Check the config is created
    assert isinstance(Meta.config, dict) is True
    assert Meta.config["meta_config"] == {"seed": 0, "verbose": True, "log_path": None}

    # Remove the temp folder
    shutil.rmtree(dirpath)
