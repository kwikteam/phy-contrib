# -*- coding: utf-8 -*-

"""Kwik GUI."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
import os.path as op

import click

from phy import IPlugin, get_plugin, load_master_config
from phy.cluster.manual import (ManualClustering, WaveformView,
                                default_wizard_functions,
                                )
from phy.gui import GUI, create_app, run_app
from phy.io.context import Context
from phy.utils import Bunch

from phycontrib.kwik import KwikModel

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Kwik GUI
#------------------------------------------------------------------------------

def attach_plugins(gui, plugins=None, session=None):
    """Attach a list of plugins to a GUI.

    By default, the list of plugins is taken from the `c.TheGUI.plugins`
    parameter, where `TheGUI` is the name of the GUI class.

    """
    session = session or {}

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
        get_plugin(plugin)().attach_to_gui(gui, session)


class KwikGUI(GUI):
    """Manual clustering GUI working with Kwik datasets."""

    def __init__(self, path, plugins=None):
        # Initialize the GUI.
        super(KwikGUI, self).__init__()

        # Load the Kwik dataset.
        self.path = op.realpath(op.expanduser(path))
        self.model = KwikModel(path)

        ctx = Context(op.join(op.dirname(self.path), '.phy/'))

        # Attach the manual clustering logic (wizard, merge, split,
        # undo stack) to the GUI.
        mc = ManualClustering(self.model.spike_clusters,
                              cluster_groups=self.model.cluster_groups,
                              n_spikes_max_per_cluster=100,  # TODO
                              )
        mc.attach(self)

        spc = self.model.spikes_per_cluster
        nfc = self.model.n_features_per_channel

        q, s = default_wizard_functions(waveforms=self.model.waveforms,
                                        features=self.model.features,
                                        masks=self.model.masks,
                                        n_features_per_channel=nfc,
                                        spikes_per_cluster=spc,
                                        )

        mc.set_quality_func(ctx.cache(q))
        mc.set_similarity_func(ctx.cache(s))

        # Create the waveform view.
        w = WaveformView(waveforms=self.model.waveforms,
                         masks=self.model.masks,
                         spike_clusters=self.model.spike_clusters,
                         channel_positions=self.model.probe.positions,
                         keys=None,  # disable Escape shortcut in the view
                         )
        w.attach(self)

        # Create the context to pass to the plugins in `attach_to_gui()`.
        session = Bunch({
            'path': path,
            'model': self.model,
            'manual_clustering': mc,
            'context': ctx,
        })

        # Attach the specified plugins.
        attach_plugins(self, plugins, session)


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
            gui = KwikGUI(path, plugins=['SaveGUIState'])

            # Show the GUI.
            gui.show()

            # Start the Qt event loop.
            run_app()

            # Close the GUI.
            gui.close()
            del gui


class SaveGUIState(IPlugin):
    def attach_to_gui(self, gui, session):

        gs_name = '{}/geometry_state'.format(gui.name)

        @gui.connect_
        def on_close():
            gs = gui.save_geometry_state()
            logger.debug("Save geometry state to %s.", gs_name)
            session.context.save(gs_name, gs)

        @gui.connect_
        def on_show():
            gs = session.context.load(gs_name)
            logger.debug("Restore geometry state from %s.", gs_name)
            gui.restore_geometry_state(gs)
