# -*- coding: utf-8 -*-

"""Kwik GUI."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
import os.path as op
import shutil

import click

from phy import IPlugin
from phy.cluster.manual import (ManualClustering,
                                WaveformView,
                                CorrelogramView,
                                TraceView,
                                default_wizard_functions,
                                )
from phy.gui import GUI, create_app, run_app, load_gui_plugins
from phy.io.context import Context
from phy.utils import Bunch

from phycontrib.kwik import KwikModel

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Kwik GUI
#------------------------------------------------------------------------------

class KwikGUI(GUI):
    """Manual clustering GUI working with Kwik datasets."""

    def __init__(self, path, plugins=None):
        # Initialize the GUI.
        super(KwikGUI, self).__init__()

        # Load the Kwik dataset.
        self.path = op.realpath(op.expanduser(path))

        # Backup the kwik file.
        path_backup = self.path + '.bak'
        if not op.exists(path_backup):
            logger.info("Backup `%s`.".format(path_backup))
            shutil.copy(self.path, path_backup)

        # Initialize the Kwik model.
        self.model = KwikModel(path)

        # Create the computing context.
        ctx = Context(op.join(op.dirname(self.path), '.phy/'))

        # Attach the manual clustering logic (wizard, merge, split,
        # undo stack) to the GUI.
        mc = ManualClustering(self.model.spike_clusters,
                              cluster_groups=self.model.cluster_groups,
                              n_spikes_max_per_cluster=100,  # TODO
                              )
        mc.attach(self)

        spc = mc.clustering.spikes_per_cluster
        nfc = self.model.n_features_per_channel

        q, s = default_wizard_functions(waveforms=self.model.waveforms,
                                        features=self.model.features,
                                        masks=self.model.masks,
                                        n_features_per_channel=nfc,
                                        spikes_per_cluster=spc,
                                        )

        mc.add_column(ctx.cache(q), name='quality')
        mc.set_default_sort('quality')
        mc.set_similarity_func(ctx.cache(s))

        # Create the waveform view.
        w = WaveformView(waveforms=self.model.waveforms,
                         masks=self.model.masks,
                         spike_clusters=self.model.spike_clusters,
                         channel_positions=self.model.probe.positions,
                         )
        w.attach(self)

        # Create the waveform view.
        ccg = CorrelogramView(spike_times=self.model.spike_times,
                              spike_clusters=self.model.spike_clusters,
                              sample_rate=self.model.sample_rate,
                              bin_size=1e-3,
                              window_size=50e-3,
                              )
        ccg.attach(self)

        tv = TraceView(traces=self.model.traces,
                       sample_rate=self.model.sample_rate,
                       spike_times=self.model.spike_times,
                       spike_clusters=self.model.spike_clusters,
                       masks=self.model.masks,
                       )
        tv.attach(self)

        @self.connect_
        def on_request_save(spike_clusters, groups):
            groups = {c: g.title() for c, g in groups.items()}
            self.model.save(spike_clusters, groups)

        # Create the context to pass to the plugins in `attach_to_gui()`.
        session = Bunch({
            'path': path,
            'model': self.model,
            'manual_clustering': mc,
            'context': ctx,
        })

        # Attach the specified plugins.
        load_gui_plugins(self, plugins, session)


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
