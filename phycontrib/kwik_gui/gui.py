# -*- coding: utf-8 -*-

"""Kwik GUI."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging

import click

from phy import IPlugin, get_plugin, load_master_config
from phy.cluster.manual import ManualClustering, WaveformView
from phy.gui import GUI, create_app, run_app

from phycontrib.kwik import KwikModel

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Kwik GUI
#------------------------------------------------------------------------------

def attach_plugins(gui, plugins=None, ctx=None):
    """Attach a list of plugins to a GUI.

    By default, the list of plugins is taken from the `c.TheGUI.plugins`
    parameter, where `TheGUI` is the name of the GUI class.

    """
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
    """Manual clustering GUI working with Kwik datasets."""

    def __init__(self, path, plugins=None):
        # Initialize the GUI.
        super(KwikGUI, self).__init__()

        # Load the Kwik dataset.
        self.path = path
        self.model = KwikModel(path)

        # Attach the manual clustering logic (wizard, merge, split,
        # undo stack) to the GUI.
        mc = ManualClustering(self.model.spike_clusters,
                              cluster_groups=self.model.cluster_groups,
                              n_spikes_max_per_cluster=100,  # TODO
                              )
        mc.attach(self)

        # Create the waveform view.
        w = WaveformView(waveforms=self.model.waveforms,
                         masks=self.model.masks,
                         spike_clusters=self.model.spike_clusters,
                         channel_positions=self.model.probe.positions,
                         )
        w.attach(self)

        # Create the context to pass to the plugins in `attach_to_gui()`.
        ctx = {
            'path': path,
            'model': self.model,
            'manual_clustering': mc,
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

            # Create the Qt application.
            create_app()

            # Create the Kwik GUI.
            gui = KwikGUI(path)

            # Show the GUI.
            gui.show()

            # Start the Qt event loop.
            run_app()
