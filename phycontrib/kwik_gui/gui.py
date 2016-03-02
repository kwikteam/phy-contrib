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
from phy.gui import create_app, run_app
from phy.cluster.manual.controller import Controller

from phycontrib.kwik import KwikModel

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Kwik GUI
#------------------------------------------------------------------------------

def _backup(path):
    """Backup a file."""
    path_backup = path + '.bak'
    if not op.exists(path_backup):
        logger.info("Backup `%s`.".format(path_backup))
        shutil.copy(path, path_backup)


class KwikController(Controller):
    def __init__(self, path):
        path = op.realpath(op.expanduser(path))
        _backup(path)
        self.cache_dir = op.join(op.dirname(path), '.phy')
        self.model = KwikModel(path)
        super(KwikController, self).__init__()

    def _init_data(self):
        m = self.model
        self.spike_times = m.spike_times

        self.spike_clusters = m.spike_clusters
        self.cluster_groups = m.cluster_groups
        self.cluster_ids = m.cluster_ids

        self.channel_positions = m.channel_positions
        self.n_samples_waveforms = m.n_samples_waveforms
        self.n_channels = m.n_channels
        self.n_features_per_channel = m.n_features_per_channel
        self.sample_rate = m.sample_rate
        self.duration = m.duration

        self.all_masks = m.all_masks
        self.all_waveforms = m.all_waveforms
        self.all_features = m.all_features
        self.all_traces = m.all_traces

    def create_gui(self, plugins=None, config_dir=None):
        """Create the kwik GUI."""
        create = super(KwikController, self).create_gui
        gui = create(name='KwikGUI', subtitle=self.path,
                     plugins=plugins, config_dir=config_dir)
        model = self.model

        # Save.
        @gui.connect_
        def on_request_save(spike_clusters, groups):
            groups = {c: g.title() for c, g in groups.items()}
            model.save(spike_clusters, groups)

        return gui


#------------------------------------------------------------------------------
# Kwik GUI plugin
#------------------------------------------------------------------------------

class KwikGUIPlugin(IPlugin):
    """Create the `phy cluster-manual` command for Kwik files."""

    def attach_to_cli(self, cli):

        # Create the `phy cluster-manual file.kwik` command.
        @cli.command('kwik-gui')
        @click.argument('path', type=click.Path(exists=True))
        def cluster_manual(path):

            # Create the Qt application.
            create_app()

            controller = KwikController(path)
            gui = controller.create_gui()

            gui.show()
            run_app()
            gui.close()
            del gui
