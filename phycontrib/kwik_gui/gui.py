# -*- coding: utf-8 -*-

"""Kwik GUI."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging

import click

from phy import IPlugin, get_plugin, load_master_config
from phy.cluster.manual import ManualClustering
from phy.gui import Actions, GUI, create_app, run_app

from phycontrib.kwik import KwikModel

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Kwik GUI
#------------------------------------------------------------------------------

def attach_plugins(gui, plugins, ctx=None):
    ctx = ctx or {}

    # GUI name.
    name = gui.__class__.__name__

    # If no plugins are specified, load the master config and
    # get the list of user plugins to attach to the GUI.
    if plugins is None:
        config = load_master_config()
        plugins = config[name].plugins
        if not isinstance(plugins, list):
            plugins = []

    # Attach the plugins to the GUI.
    for plugin in plugins:
        logger.info("Attach plugin `%s` to %s.", plugin, name)
        get_plugin(plugin)().attach_to_gui(gui, ctx)


class KwikGUI(GUI):
    def __init__(self, path, plugins=None):
        # Initialize the GUI.
        super(KwikGUI, self).__init__()

        # Initialize the actions.
        self.actions = Actions(self)

        # Load the Kwik dataset.
        self.path = path
        self.model = KwikModel(path)

        # Create the context to pass to the plugins in `attach_to_gui()`.
        ctx = {
            'path': path,
        }

        # Attach the specified plugins.
        attach_plugins(self, plugins, ctx)


#------------------------------------------------------------------------------
# Kwik GUI plugin
#------------------------------------------------------------------------------

class KwikGUIPlugin(IPlugin):
    """Create the `phy cluster-manual` command for Kwik files."""

    def attach_to_cli(self, cli):

        # Create the `phy cluster-manual file.kwik` command.
        @cli.command('cluster-manual')
        @click.argument('path', type=click.Path(exists=True))
        def cluster_manual(path):
            create_app()
            gui = KwikGUI(path)
            gui.show()
            run_app()
