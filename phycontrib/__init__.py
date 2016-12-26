# -*- coding: utf-8 -*-

"""phycontrib."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
import os
import os.path as op
import shutil

from .kwik import KwikGUIPlugin, KwikController  # noqa
from .template import TemplateGUIPlugin, TemplateController  # noqa
from . import utils  # noqa

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Default config and state files
#------------------------------------------------------------------------------

__version__ = '1.0.15'


def _copy_gui_state(gui_name, module_name, config_dir=None):
    """Copy the state.json file."""
    config_dir = config_dir or op.join(op.realpath(op.expanduser('~')), '.phy')
    gui_dir = op.join(config_dir, gui_name)
    if not op.exists(gui_dir):
        os.makedirs(gui_dir)
    # Create the script if it doesn't already exist.
    path = op.join(gui_dir, 'state.json')
    if op.exists(path):
        return
    curdir = op.dirname(op.realpath(__file__))
    from_path = op.join(curdir, module_name, 'static', 'state.json')
    logger.debug("Copy %s to %s" % (from_path, path))
    shutil.copy(from_path, path)


def _copy_all_gui_states():
    _copy_gui_state('KwikGUI', 'kwik')
    _copy_gui_state('TemplateGUI', 'template')


# Copy default states when importing the package.
_copy_all_gui_states()
