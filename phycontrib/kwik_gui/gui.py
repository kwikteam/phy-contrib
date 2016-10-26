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
from phy.cluster.manual.controller import Controller
from phy.gui import create_app, run_app
from phy.utils.tempdir import TemporaryDirectory
from phy.utils.cli import _run_cmd, _add_log_file

logger = logging.getLogger(__name__)

try:
    from klusta.kwik import KwikModel
    from klusta.launch import cluster
except ImportError:
    logger.warn("Package klusta not installed: the KwikGUI will not work.")


#------------------------------------------------------------------------------
# Kwik GUI
#------------------------------------------------------------------------------

def _backup(path):
    """Backup a file."""
    path_backup = path + '.bak'
    if not op.exists(path_backup):
        logger.info("Backup `{0}`.".format(path_backup))
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
        self.channel_order = m.channel_order
        self.n_samples_waveforms = m.n_samples_waveforms
        self.n_channels = m.n_channels
        self.n_features_per_channel = m.n_features_per_channel
        self.sample_rate = m.sample_rate
        self.duration = m.duration

        self.all_masks = m.all_masks
        self.all_waveforms = m.all_waveforms
        self.all_features = m.all_features
        # WARNING: m.all_traces contains the dead channels, m.traces doesn't.
        # Also, m.traces has the reordered channels as per the prb file.
        self.all_traces = m.traces

    def create_gui(self, config_dir=None):
        """Create the kwik GUI."""
        f = super(KwikController, self).create_gui
        gui = f(name=self.gui_name,
                subtitle=self.path,
                config_dir=config_dir,
                )

        @self.manual_clustering.actions.add
        def recluster():
            """Relaunch KlustaKwik on the selected clusters."""
            # Selected clusters.
            cluster_ids = self.manual_clustering.selected
            spike_ids = self.selector.select_spikes(cluster_ids)
            logger.info("Running KlustaKwik on %d spikes.", len(spike_ids))

            # Run KK2 in a temporary directory to avoid side effects.
            with TemporaryDirectory() as tempdir:
                spike_clusters, metadata = cluster(self.model,
                                                   spike_ids,
                                                   num_starting_clusters=10,
                                                   tempdir=tempdir,
                                                   )
            self.manual_clustering.split(spike_ids, spike_clusters)

        # Save.
        @gui.connect_
        def on_request_save(spike_clusters, groups):
            groups = {c: g.title() for c, g in groups.items()}
            self.model.save(spike_clusters, groups)

        return gui


#------------------------------------------------------------------------------
# Kwik GUI plugin
#------------------------------------------------------------------------------

def _run(path, channel_group, clustering):
    controller = KwikController(path,
                                channel_group=channel_group,
                                clustering=clustering,
                                )
    gui = controller.create_gui()
    gui.show()
    run_app()
    gui.close()
    del gui


class KwikGUIPlugin(IPlugin):
    """Create the `phy cluster-manual` command for Kwik files."""

    def attach_to_cli(self, cli):

        # Create the `phy cluster-manual file.kwik` command.
        @cli.command('kwik-gui')
        @click.argument('path', type=click.Path(exists=True))
        @click.option('--channel-group', type=int)
        @click.option('--clustering', type=str)
        @click.pass_context
        def cluster_manual(ctx, path, channel_group=None, clustering=None):
            """Launch the Kwik GUI on a Kwik file."""

            # Create a `phy.log` log file with DEBUG level.
            _add_log_file(op.join(op.dirname(path), 'phy.log'))

            create_app()
            _run_cmd('_run(path, channel_group, clustering)',
                     ctx, globals(), locals())

        @cli.command('kwik-describe')
        @click.argument('path', type=click.Path(exists=True))
        @click.option('--channel-group', type=int,
                      help='channel group')
        @click.option('--clustering', type=str,
                      help='clustering')
        def describe(path, channel_group=0, clustering='main'):
            """Describe a Kwik file."""
            KwikModel(path,
                      channel_group=channel_group,
                      clustering=clustering,
                      ).describe()
