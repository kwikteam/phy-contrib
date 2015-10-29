# -*- coding: utf-8 -*-

"""Kwik GUI."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

# import sys

import click
# import numpy as np

from phy import IPlugin
# from phy.cluster.manual import ManualClustering
from phy.gui import GUI, create_app, run_app


#------------------------------------------------------------------------------
# Kwik GUI
#------------------------------------------------------------------------------

class KwikGUI(GUI):
    def __init__(self, path):
        super(KwikGUI, self).__init__()
        self.path = path
        # TODO: load plugins with attach_to_gui(gui, ctx)
        # model = KwikModel(path)
        # config = load_master_config()
        # plugins = config.KwikGUI.plugins
        # attach_plugins(gui, plugins)


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
