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

logger = logging.getLogger(__name__)

try:
    from klusta.kwik import KwikModel
except ImportError:
    logger.warn("Package klusta not installed: the KwikGUI will not work.")


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
    gui_name = 'KwikGUI'

    def __init__(self, path, channel_group=None, clustering=None):
        path = op.realpath(op.expanduser(path))
        _backup(path)
        self.path = path
        self.cache_dir = op.join(op.dirname(path), '.phy',
                                 str(clustering or 'main'),
                                 str(channel_group or 'default'),
                                 )
        self.model = KwikModel(path,
                               channel_group=channel_group,
                               clustering=None,
                               )
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
        gui = create(name=self.gui_name, subtitle=self.path,
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
        @click.option('--channel-group', type=int)
        @click.option('--clustering', type=str)
        def cluster_manual(path, channel_group=None, clustering=None):
            """Launch the Kwik GUI on a Kwik file."""

            # Create the Qt application.
            create_app()

            controller = KwikController(path,
                                        channel_group=channel_group,
                                        clustering=clustering,
                                        )
            gui = controller.create_gui()

            gui.show()
            run_app()
            gui.close()
            del gui

        @cli.command('kwik-describe')
        @click.argument('path', type=click.Path(exists=True))
        def describe(path):
            """Describe a Kwik file."""
            KwikModel(path).describe()
