
# import sys

import click
# import numpy as np

from phy import IPlugin
# from phy.cluster.manual import ManualClustering
from phy.gui import GUI, create_app, run_app


class KwikGUI(GUI):
    def __init__(self, path):
        self.path = path
        # TODO
        # model = KwikModel(path)
        # config = load_master_config()
        # plugins = config.KwikGUI.plugins
        # attach_plugins(gui, plugins)


class KwikGUIPlugin(IPlugin):
    def attach_to_cli(self, cli):
        @cli.command('cluster-manual')
        @click.argument('path', type=click.Path(exists=True))
        def cluster_manual(path):
            create_app()
            gui = KwikGUI(path)
            # TODO: load plugins with attach_to_gui(gui, ctx)

            gui.show()
            run_app()
