# -*- coding: utf-8 -*-

"""Kwik GUI."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging

import click

from phy import IPlugin
from phy.gui import create_app, create_gui, run_app
from phy.cluster.manual.gui_component import ManualClustering
from phy.cluster.manual.views import (WaveformView,
                                      TraceView,
                                      FeatureView,
                                      CorrelogramView,
                                      )

from phycontrib.kwik import create_model

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Kwik GUI
#------------------------------------------------------------------------------

def add_waveform_view(gui):
    model = gui.model
    v = WaveformView(waveforms=model.waveforms,
                     channel_positions=model.channel_positions,
                     n_samples=model.n_samples_waveforms,
                     waveform_lim=model.waveform_lim(),
                     best_channels=model.best_channels_multiple,
                     )
    v.attach(gui)
    return v


def add_trace_view(gui):
    model = gui.model
    v = TraceView(traces=model.traces,
                  spikes=model.spikes_traces,
                  sample_rate=model.sample_rate,
                  duration=model.duration,
                  n_channels=model.n_channels,
                  )
    v.attach(gui)
    return v


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
    return v


def add_correlogram_view(gui):
    model = gui.model
    v = CorrelogramView(spike_times=model.spike_times,
                        spike_clusters=model.spike_clusters,
                        sample_rate=model.sample_rate,
                        )
    v.attach(gui)
    return v


def create_kwik_gui(path, plugins=None):
    # Backup the kwik file, create the model, context, and selector.
    model = create_model(path)

    # Create the GUI.
    gui = create_gui(name='KwikGUI',
                     subtitle=model.kwik_path,
                     plugins=plugins,
                     )
    gui.model = model

    # Create the manual clustering.
    mc = ManualClustering(model.spike_clusters,
                          model.spikes_per_cluster,
                          cluster_groups=model.cluster_groups,)
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
