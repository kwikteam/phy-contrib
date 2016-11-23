# -*- coding: utf-8 -*-

"""Template GUI."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
from operator import itemgetter
import os.path as op

import click
import numpy as np

from phy.cluster.picker import ClusterPicker
from phy.cluster.views import (WaveformView,
                               FeatureView,
                               # CorrelogramView,
                               # TraceView,
                               # ScatterView,
                               )
from phy.gui import create_app, run_app, GUI
from phy.io.array import (Selector,
                          )
from phy.io.context import Context
from phy.utils.cli import _run_cmd
from phy.utils import Bunch, IPlugin, EventEmitter
from phy.utils.cli import _add_log_file
from phy.utils._misc import _read_python

from .model import TemplateModel

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Utils
#------------------------------------------------------------------------------

def _dat_n_samples(filename, dtype=None, n_channels=None, offset=None):
    assert dtype is not None
    item_size = np.dtype(dtype).itemsize
    offset = offset if offset else 0
    n_samples = (op.getsize(filename) - offset) // (item_size * n_channels)
    assert n_samples >= 0
    return n_samples


def _dat_to_traces(dat_path, n_channels=None, dtype=None, offset=None):
    assert dtype is not None
    assert n_channels is not None
    n_samples = _dat_n_samples(dat_path,
                               n_channels=n_channels,
                               dtype=dtype,
                               offset=offset,
                               )
    return np.memmap(dat_path, dtype=dtype, shape=(n_samples, n_channels),
                     offset=offset)


#------------------------------------------------------------------------------
# Template views
#------------------------------------------------------------------------------

def subtract_templates(traces,
                       start=None,
                       spike_times=None,
                       spike_clusters=None,
                       amplitudes=None,
                       spike_templates=None,
                       sample_rate=None,
                       ):
    traces = traces.copy()
    st = spike_times
    w = spike_templates
    n_spikes, n_samples_t, n_channels = w.shape
    n = traces.shape[0]
    for index in range(w.shape[0]):
        t = int(round((st[index] - start) * sample_rate))
        i, j = n_samples_t // 2, n_samples_t // 2 + (n_samples_t % 2)
        assert i + j == n_samples_t
        x = w[index] * amplitudes[index]  # (n_samples, n_channels)
        sa, sb = t - i, t + j
        if sa < 0:
            x = x[-sa:, :]
            sa = 0
        elif sb > n:
            x = x[:-(sb - n), :]
            sb = n
        traces[sa:sb, :] -= x
    return traces


#------------------------------------------------------------------------------
# Template Controller
#------------------------------------------------------------------------------

class TemplateController(EventEmitter):
    gui_name = 'TemplateGUI'

    n_spikes_waveforms = 100
    batch_size_waveforms = 10

    n_spikes_features = 10000

    def __init__(self, dat_path, config_dir=None, **kwargs):
        super(TemplateController, self).__init__()
        dat_path = op.realpath(dat_path)
        self.model = TemplateModel(dat_path, **kwargs)
        self.cache_dir = op.join(self.model.dir_path, '.phy')
        self.context = Context(self.cache_dir)
        self.config_dir = config_dir

        self._set_picker()
        self._set_selector()

    def _set_picker(self):
        # Load the new cluster id.
        new_cluster_id = self.context.load('new_cluster_id'). \
            get('new_cluster_id', None)
        cluster_groups = self.model.metadata['group']
        self.picker = ClusterPicker(self.model.spike_clusters,
                                    similarity=self.similarity,
                                    cluster_groups=cluster_groups,
                                    new_cluster_id=new_cluster_id,
                                    )
        self.picker.add_column(self.get_best_channel, name='channel')
        self.picker.add_column(self.get_probe_depth, name='depth')

    def _set_selector(self):
        def spikes_per_cluster(cluster_id):
            return self.picker.clustering.spikes_per_cluster[cluster_id]
        self.selector = Selector(spikes_per_cluster)

    def get_template_counts(self, cluster_id):
        """Return a histogram of the number of spikes in each template for
        a given cluster."""
        spike_ids = self.picker.clustering.spikes_per_cluster[cluster_id]
        st = self.model.spike_templates[spike_ids]
        return np.bincount(st, minlength=self.model.n_templates)

    def get_template_for_cluster(self, cluster_id):
        """Return the template associated to each cluster."""
        spike_ids = self.picker.clustering.spikes_per_cluster[cluster_id]
        st = self.model.spike_templates[spike_ids]
        template_ids, counts = np.unique(st, return_counts=True)
        ind = np.argmax(counts)
        return template_ids[ind]

    def similarity(self, cluster_id):
        """Return the list of similar clusters to a given cluster."""
        # Templates of the cluster.
        temp_i = np.nonzero(self.get_template_counts(cluster_id))[0]
        # The similarity of the cluster with each template.
        sims = np.max(self.model.similar_templates[temp_i, :], axis=0)

        def _sim_ij(cj):
            # Templates of the cluster.
            if cj < self.model.n_templates:
                return sims[cj]
            temp_j = np.nonzero(self.get_template_counts(cj))[0]
            return np.max(sims[temp_j])

        out = [(cj, _sim_ij(cj)) for cj in self.picker.clustering.cluster_ids]
        return sorted(out, key=itemgetter(1), reverse=True)

    def get_best_channel(self, cluster_id):
        """Return the best channel of a given cluster."""
        template_id = self.get_template_for_cluster(cluster_id)
        return self.model.get_template(template_id).best_channel

    def get_best_channels(self, cluster_id):
        """Return the best channels of a given cluster."""
        template_id = self.get_template_for_cluster(cluster_id)
        return self.model.get_template(template_id).channels

    def get_probe_depth(self, cluster_id):
        """Return the depth of a cluster."""
        channel_id = self.get_best_channel(cluster_id)
        return self.model.channel_positions[channel_id][1]

    def _add_view(self, gui, view):
        view.attach(gui)
        self.emit('add_view', gui, view)
        return view

    def _get_waveforms(self, cluster_id):
        spike_ids = self.selector.select_spikes([cluster_id],
                                                self.n_spikes_waveforms,
                                                self.batch_size_waveforms,
                                                )
        channel_ids = self.get_best_channels(cluster_id)
        data = self.model.get_waveforms(spike_ids, channel_ids)
        return Bunch(data=data,
                     spike_ids=spike_ids,
                     channel_ids=channel_ids,
                     )

    def add_waveform_view(self, gui):
        v = WaveformView(waveforms=self._get_waveforms,
                         channel_positions=self.model.channel_positions,
                         channel_order=self.model.channel_mapping,
                         best_channels=self.get_best_channels,
                         )
        return self._add_view(gui, v)

    def _get_features(self, cluster_id):
        spike_ids = self.selector.select_spikes([cluster_id],
                                                self.n_spikes_features,
                                                )
        channel_ids = self.get_best_channels(cluster_id)
        data = self.model.get_features(spike_ids, channel_ids)
        return Bunch(data=data,
                     spike_ids=spike_ids,
                     channel_ids=channel_ids,
                     )

    def add_feature_view(self, gui):
        nfpc = self.model.n_features_per_channel
        v = FeatureView(features=self._get_features,
                        spike_times=self.model.spike_times,
                        n_channels=self.model.n_channels,
                        n_features_per_channel=nfpc,
                        best_channels=self.get_best_channels,
                        )
        return self._add_view(gui, v)

    def create_gui(self, **kwargs):
        gui = GUI(name=self.gui_name,
                  subtitle=self.model.dat_path,
                  config_dir=self.config_dir,
                  **kwargs)

        self.picker.attach(gui)

        self.add_waveform_view(gui)
        self.add_feature_view(gui)

        return gui


#------------------------------------------------------------------------------
# Template GUI plugin
#------------------------------------------------------------------------------

def _run(params):
    controller = TemplateController(**params)
    gui = controller.create_gui()
    gui.show()
    run_app()
    gui.close()
    del gui


class TemplateGUIPlugin(IPlugin):
    """Create the `phy template-gui` command for Kwik files."""

    def attach_to_cli(self, cli):

        # Create the `phy cluster-manual file.kwik` command.
        @cli.command('template-gui')
        @click.argument('params-path', type=click.Path(exists=True))
        @click.pass_context
        def cluster_manual(ctx, params_path):
            """Launch the Template GUI on a params.py file."""

            # Create a `phy.log` log file with DEBUG level.
            _add_log_file(op.join(op.dirname(params_path), 'phy.log'))

            create_app()

            params = _read_python(params_path)
            params['dtype'] = np.dtype(params['dtype'])

            _run_cmd('_run(params)', ctx, globals(), locals())
