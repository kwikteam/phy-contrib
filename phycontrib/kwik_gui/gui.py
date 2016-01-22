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
from phy.gui import create_app, create_gui, run_app
from phy.io import Context, Selector
from phy.cluster.manual.gui_component import ManualClustering
from phy.cluster.manual.views import (WaveformView,
                                      TraceView,
                                      FeatureView,
                                      CorrelogramView,
                                      )

from phycontrib.kwik import KwikModel, create_cluster_store

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


def add_waveform_view(gui):
    model = gui.model
    v = WaveformView(waveforms=model.waveforms,
                     channel_positions=model.channel_positions,
                     n_samples=model.n_samples_waveforms,
                     waveform_lim=model.waveform_lim(),
                     best_channels=model.best_channels_multiple,
                     )
    v.attach(gui)


def add_trace_view(gui):
    model = gui.model
    v = TraceView(traces=model.traces,
                  sample_rate=model.sample_rate,
                  n_samples_per_spike=model.n_samples_waveforms,
                  duration=model.duration,
                  n_channels=model.n_channels,
                  )
    v.attach(gui)


def add_feature_view(gui):
    model = gui.model
    v = FeatureView(features=model.features,
                    background_features=model.background_features(),
                    spike_times=model.spike_times,
                    n_channels=model.n_channels,
                    n_features_per_channel=model.n_features_per_channel,
                    feature_lim=model.feature_lim(),
                    )
    v.attach(gui)


def add_correlogram_view(gui):
    model = gui.model
    v = CorrelogramView(spike_times=model.spike_times,
                        spike_clusters=model.spike_clusters,
                        sample_rate=model.sample_rate,
                        )
    v.attach(gui)


def create_kwik_gui(path, plugins=None):
    # Open the dataset.
    path = op.realpath(op.expanduser(path))
    _backup(path)
    model = KwikModel(path)

    # Create the GUI.
    gui = create_gui(name='KwikGUI',
                     subtitle=model.kwik_path,
                     model=model,
                     plugins=plugins,
                     )

    # Create the manual clustering.
    mc = ManualClustering(model.spike_clusters,
                          cluster_groups=model.cluster_groups,)

    # Create the context.
    context = Context(op.join(op.dirname(path), '.phy'))

    # Create the store.
    def spikes_per_cluster(cluster_id):
        # HACK: we get the spikes_per_cluster from the Clustering instance.
        # We need to access it from a function to avoid circular dependencies
        # between the cluster store and manual clustering plugins.
        mc = gui.request('manual_clustering')
        return mc.clustering.spikes_per_cluster[cluster_id]

    selector = Selector(spike_clusters=model.spike_clusters,
                        spikes_per_cluster=spikes_per_cluster,
                        )
    create_cluster_store(model, selector=selector, context=context)
    mc.attach(gui)
    gui.manual_clustering = mc

    # Add the views.
    add_waveform_view(gui)
    add_feature_view(gui)
    add_trace_view(gui)
    add_correlogram_view(gui)

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
        @cli.command('cluster-manual')
        @click.argument('path', type=click.Path(exists=True))
        def cluster_manual(path):

            # Create the Qt application.
            create_app()

            # List of plugins activated by default.
            plugins = ['SaveGeometryStatePlugin',
                       ]

            gui = create_kwik_gui(path, plugins=plugins)

            # Show the GUI.
            gui.show()

            # Start the Qt event loop.
            run_app()

            # Close the GUI.
            gui.close()
            del gui
