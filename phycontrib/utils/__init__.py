# -*- coding: utf-8 -*-

"""Utils."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging

from phy.utils import load_master_config, get_plugin

from .precache import PrecachePlugin  # noqa
from .saveprompt import SavePromptPlugin  # noqa

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Functions
#------------------------------------------------------------------------------

def attach_plugins(controller, plugins=None, config_dir=None):
    # Attach the plugins.
    plugins = plugins or []
    config = load_master_config(config_dir=config_dir)
    c = config.get(controller.gui_name or controller.__class__.__name__)
    default_plugins = c.plugins if c else []
    if len(default_plugins):
        plugins = default_plugins + plugins
    for plugin in plugins:
        try:
            p = get_plugin(plugin)()
        except ValueError:  # pragma: no cover
            logger.warn("The plugin %s couldn't be found.", plugin)
            continue
        try:
            p.attach_to_controller(controller)
            logger.debug("Attached plugin %s.", plugin)
        except Exception as e:  # pragma: no cover
            logger.warn("An error occurred when attaching plugin %s: %s.",
                        plugin, e)
