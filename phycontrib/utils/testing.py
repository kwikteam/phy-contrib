# -*- coding: utf-8 -*-

"""Testing utility functions."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
import os.path as op

from phy import download_file
from phy.utils import phy_config_dir, _ensure_dir_exists

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Utility functions
#------------------------------------------------------------------------------

_BASE_URL = 'https://raw.githubusercontent.com/kwikteam/phy-data/master/'


def download_test_file(name, config_dir=None, force=False):
    """Download a test file."""
    config_dir = config_dir or phy_config_dir()
    path = op.join(config_dir, 'test_data', name)
    _ensure_dir_exists(op.dirname(path))
    if not force and op.exists(path):
        return path
    url = _BASE_URL + name
    download_file(url, output_path=path)
    return path
