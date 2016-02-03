# -*- coding: utf-8 -*-

"""Template GUI."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
import os.path as op

import numpy as np

from phy.cluster.manual import ManualClustering
from phy.cluster.manual.controller import Controller
from phy.cluster.manual.views import (select_traces, ScatterView)

from phy.gui import create_gui
from phy.io.array import concat_per_cluster
from phy.utils import Bunch

from .model import TemplateModel

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Template Controller
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
    w = spike_templates * amplitudes[:, np.newaxis, np.newaxis]
    n = traces.shape[0]
    for index in range(w.shape[0]):
        t = int(round((st[index] - start) * sample_rate))
        i, j = 20, 41
        x = w[index]  # (n_samples, n_channels)
        sa, sb = t - i, t + j
        if sa < 0:
            x = x[-sa:, :]
            sa = 0
        elif sb > n:
            x = x[:-(sb - n), :]
            sb = n
        traces[sa:sb, :] -= x
    return traces


class AmplitudeView(ScatterView):
    pass


class FeatureTemplateView(ScatterView):
    pass


class TemplateController(Controller):
    def __init__(self, dat_path, **kwargs):
        path = op.realpath(op.expanduser(dat_path))
        self.cache_dir = op.join(op.dirname(path), '.phy')
        self.model = TemplateModel(dat_path, **kwargs)
        super(TemplateController, self).__init__()

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

        self.templates_unw = m.templates_unw

        self.all_masks = m.all_masks
        self.all_waveforms = m.all_waveforms
        self.all_features = m.all_features
        self.all_traces = m.all_traces
        self.all_amplitudes = m.all_amplitudes

    def get_background_features(self):
        pass

    def _init_context(self):
        super(TemplateController, self)._init_context()
        ctx = self.context
        self.get_amplitudes = ctx.cache(self.get_amplitudes)
        self.get_template_features = ctx.cache(self.get_template_features)

    @concat_per_cluster
    def get_features(self, cluster_id):
        m = self.model
        spike_ids = self._select_spikes(cluster_id, 1000)
        nc = self.n_channels
        nfpc = self.n_features_per_channel
        ns = len(spike_ids)
        shape = (ns, nc, nfpc)
        f = np.zeros(shape)
        # Sparse channels.
        ch = m.features_ind[:, cluster_id]
        # Populate the dense features array.
        f[:, ch, :] = np.transpose(m.all_features[spike_ids, :, :], (0, 2, 1))
        m = self.get_masks(cluster_id).data

        b = Bunch()
        b.data = f
        b.spike_ids = spike_ids
        b.spike_clusters = self.spike_clusters[spike_ids]
        b.masks = self.all_masks[spike_ids]
        return b

    @concat_per_cluster
    def get_amplitudes(self, cluster_id):
        spike_ids = self._select_spikes(cluster_id, 10000)
        d = Bunch()
        d.spike_ids = spike_ids
        d.spike_clusters = cluster_id * np.ones(len(spike_ids), dtype=np.int32)
        d.x = self.spike_times[spike_ids]
        d.y = self.all_amplitudes[spike_ids]
        return d

    def get_template_features(self, cluster_ids):
        template_features = self.model.template_features
        template_features_ind = self.model.template_features_ind
        d = Bunch()
        if len(cluster_ids) < 2:
            return None
        cx, cy = map(int, cluster_ids[:2])
        sim_x = template_features_ind[cx].tolist()
        sim_y = template_features_ind[cy].tolist()
        if cx not in sim_y or cy not in sim_x:
            return None
        sxy = sim_x.index(cy)
        syx = sim_y.index(cx)
        spikes_x = self._select_spikes(cx)
        spikes_y = self._select_spikes(cy)
        spike_ids = np.hstack([spikes_x, spikes_y])
        d.x = np.hstack([template_features[spikes_x, 0],
                         template_features[spikes_y, syx]])
        d.y = np.hstack([template_features[spikes_x, sxy],
                         template_features[spikes_y, 0]])
        d.spike_ids = spike_ids
        d.spike_clusters = self.spike_clusters[spike_ids]
        return d

    def get_traces(self, interval):
        """Load traces and spikes in an interval."""
        tr = select_traces(self.all_traces, interval,
                           sample_rate=self.sample_rate,
                           )
        tr = tr - np.mean(tr, axis=0)

        a, b = self.spike_times.searchsorted(interval)
        sc = self.spike_clusters[a:b]

        # Remove templates.
        tr_sub = subtract_templates(tr,
                                    start=interval[0],
                                    spike_times=self.spike_times[a:b],
                                    spike_clusters=sc,
                                    amplitudes=self.all_amplitudes[a:b],
                                    spike_templates=self.templates_unw[sc],
                                    sample_rate=self.sample_rate,
                                    )

        return [Bunch(traces=tr),
                Bunch(traces=tr_sub, color=(.25, .25, .25, .75))]

    def similarity(self, cluster_id):
        m = self.model
        n = m.template_features_ind.shape[1]
        sim0 = m.template_features_ind[cluster_id]
        sim = [(int(c), -n + i) for i, c in enumerate(sim0)]
        sim2 = self.get_close_clusters(cluster_id)
        sim2 = [_ for _ in sim2 if _[0] not in sim0]
        sim.extend(sim2)
        return sim

    def set_manual_clustering(self, gui):
        mc = ManualClustering(self.spike_clusters,
                              self.spikes_per_cluster,
                              similarity=self.similarity,
                              cluster_groups=self.cluster_groups,
                              )
        self.manual_clustering = mc
        mc.add_column(self.get_probe_depth)
        mc.attach(gui)

    def add_amplitude_view(self, gui):
        view = AmplitudeView(coords=self.get_amplitudes,
                             )
        view.attach(gui)
        return view

    def add_feature_template_view(self, gui):
        m = self.model
        tf = m.template_features
        m = self._data_lim(tf, 100)
        view = FeatureTemplateView(coords=self.get_template_features,
                                   # data_bounds=[-m, -m, m, m],
                                   )
        view.attach(gui)
        return view


def create_template_gui(dat_path=None, plugins=None, **kwargs):
    controller = TemplateController(dat_path, **kwargs)
    # Create the GUI.
    gui = create_gui(name='TemplateGUI',
                     subtitle=dat_path,
                     plugins=plugins,
                     )
    controller.set_manual_clustering(gui)
    controller.add_waveform_view(gui)
    controller.add_feature_view(gui)
    controller.add_feature_template_view(gui)
    controller.add_amplitude_view(gui)
    controller.add_trace_view(gui)
    controller.add_correlogram_view(gui)

    # Save.
    @gui.connect_
    def on_request_save(spike_clusters, groups):
        # TODO
        pass

    return gui
