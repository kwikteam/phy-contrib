# -*- coding: utf-8 -*-

"""Template GUI."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
import os.path as op

import numpy as np
import scipy.io as sio

from phy.cluster.manual import ManualClustering
from phy.cluster.manual.controller import Controller
from phy.cluster.manual.views import (select_traces, ScatterView)

from phy.gui import create_gui
from phy.io.array import concat_per_cluster
from phy.traces import SpikeLoader, WaveformLoader
from phy.traces.filter import apply_filter, bandpass_filter
from phy.utils import Bunch

from phycontrib.kwik.model import _concatenate_virtual_arrays
from phycontrib.csicsvari.traces import read_dat

logger = logging.getLogger(__name__)


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
    w = spike_templates * amplitudes[:, np.newaxis, np.newaxis]
    n_spikes, n_samples_t, n_channels = w.shape
    n = traces.shape[0]
    for index in range(w.shape[0]):
        t = int(round((st[index] - start) * sample_rate))
        i, j = n_samples_t // 2, n_samples_t // 2 + (n_samples_t % 2)
        assert i + j == n_samples_t
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


#------------------------------------------------------------------------------
# Template Controller
#------------------------------------------------------------------------------

filenames = {
    'amplitudes': 'amplitudes.npy',
    'spike_templates': 'spike_templates.npy',
    'spike_clusters': 'spike_templates.npy',  # TODO
    'templates': 'templates.npy',
    'spike_samples': 'spike_times.npy',
    'channel_mapping': 'channel_map.npy',
    'channel_positions': 'channel_positions.npy',
    'whitening_matrix': 'whitening_mat.npy',

    'features': 'pc_features.npy',
    'features_ind': 'pc_feature_ind.npy',
    'template_features': 'template_features.npy',
    'template_features_ind': 'template_feature_ind.npy',
}


def read_array(name):
    fn = filenames[name]
    arr_name, ext = op.splitext(fn)
    if ext == '.mat':
        return sio.loadmat(fn)[arr_name]
    elif ext == '.npy':
        return np.load(fn)


def get_masks(templates):
    n_templates, n_samples_templates, n_channels = templates.shape
    templates = np.abs(templates)
    m = templates.max(axis=1)  # (n_templates, n_channels)
    mm = m.max(axis=1)  # (n_templates,
    masks = m / mm[:, np.newaxis]  # (n_templates, n_channels)
    masks[mm == 0, :] = 0
    return masks


class MaskLoader(object):
    def __init__(self, cluster_masks, spike_templates):
        self._spike_templates = spike_templates
        self._cluster_masks = cluster_masks
        self.shape = (len(spike_templates), cluster_masks.shape[1])

    def __getitem__(self, item):
        # item contains spike ids
        clu = self._spike_templates[item]
        return self._cluster_masks[clu]


def _densify(rows, arr, ind, ncols):
    ns = len(rows)
    nt = ind.shape[1]
    out = np.zeros((ns,) + arr.shape[1:-1] + (ncols,))
    out[np.arange(ns)[:, np.newaxis], ..., ind] = arr[rows[:, np.newaxis], ...,
                                                      np.arange(nt)]
    return out


class TemplateController(Controller):
    def __init__(self, dat_path=None, **kwargs):
        path = op.realpath(op.expanduser(dat_path))
        self.cache_dir = op.join(op.dirname(path), '.phy')
        self.dat_path = dat_path
        self.__dict__.update(kwargs)
        super(TemplateController, self).__init__()

    def _init_data(self):
        traces = read_dat(self.dat_path,
                          n_channels=self.n_channels_dat,
                          dtype=self.dtype or np.int16,
                          offset=self.offset,
                          )

        n_samples_t, _ = traces.shape
        assert _ == self.n_channels_dat

        amplitudes = read_array('amplitudes').squeeze()
        n_spikes, = amplitudes.shape
        self.n_spikes = n_spikes

        spike_clusters = read_array('spike_clusters').squeeze()
        spike_clusters = spike_clusters.astype(np.int32)
        assert spike_clusters.shape == (n_spikes,)
        self.spike_clusters = spike_clusters

        spike_templates = read_array('spike_templates').squeeze()
        spike_templates = spike_templates.astype(np.int32)
        assert spike_templates.shape == (n_spikes,)
        self.spike_templates = spike_templates

        spike_samples = read_array('spike_samples').squeeze()
        assert spike_samples.shape == (n_spikes,)

        templates = read_array('templates')
        templates[np.isnan(templates)] = 0
        # templates = np.transpose(templates, (2, 1, 0))
        n_templates, n_samples_templates, n_channels = templates.shape
        self.n_templates = n_templates

        channel_mapping = read_array('channel_mapping').squeeze()
        channel_mapping = channel_mapping.astype(np.int32)
        assert channel_mapping.shape == (n_channels,)

        channel_positions = read_array('channel_positions')
        assert channel_positions.shape == (n_channels, 2)

        if op.exists(filenames['features']):
            all_features = np.load(filenames['features'], mmap_mode='r')
            features_ind = read_array('features_ind').astype(np.int32)

            assert all_features.ndim == 3
            n_loc_chan = all_features.shape[2]
            assert all_features.shape == (self.n_spikes,
                                          self.n_features_per_channel,
                                          n_loc_chan,
                                          )
            # Check sparse features arrays shapes.
            assert features_ind.shape == (self.n_templates, n_loc_chan)
        else:
            all_features = None
            features_ind = None

        self.all_features = all_features
        self.features_ind = features_ind
        self.n_features_per_channel = 3

        if op.exists(filenames['template_features']):
            template_features = np.load(filenames['template_features'],
                                        mmap_mode='r')
            template_features_ind = read_array('template_features_ind'). \
                astype(np.int32)
            template_features_ind = template_features_ind.copy()
            n_sim_tem = template_features.shape[1]
            assert template_features.shape == (n_spikes, n_sim_tem)
            assert template_features_ind.shape == (n_templates, n_sim_tem)
        else:
            template_features = None
            template_features_ind = None

        self.template_features_ind = template_features_ind
        self.template_features = template_features

        self.n_channels = n_channels
        # Take dead channels into account.
        traces = _concatenate_virtual_arrays([traces], channel_mapping)

        # Amplitudes
        self.all_amplitudes = amplitudes
        self.amplitudes_lim = self.all_amplitudes.max()

        # Templates
        self.templates = templates
        self.n_samples_templates = n_samples_templates
        self.n_samples_waveforms = n_samples_templates
        self.template_lim = np.max(np.abs(self.templates))

        # Unwhiten the templates.
        self.whitening_matrix = read_array('whitening_matrix')
        self.templates_unw = np.dot(self.templates, self.whitening_matrix)

        self.duration = n_samples_t / float(self.sample_rate)

        self.spike_times = spike_samples / float(self.sample_rate)
        assert np.all(np.diff(self.spike_times) >= 0)

        self.cluster_ids = np.unique(self.spike_clusters)
        n_clusters = len(self.cluster_ids)

        self.channel_positions = channel_positions
        self.all_traces = traces

        # Filter the waveforms.
        order = 3
        filter_margin = order * 3
        b_filter = bandpass_filter(rate=self.sample_rate,
                                   low=500.,
                                   high=self.sample_rate * .475,
                                   order=order)

        def the_filter(x, axis=0):
            return apply_filter(x, b_filter, axis=axis)

        # Fetch waveforms from traces.
        nsw = self.n_samples_waveforms
        waveforms = WaveformLoader(traces=traces,
                                   n_samples_waveforms=nsw,
                                   filter=the_filter,
                                   filter_margin=filter_margin,
                                   )
        waveforms = SpikeLoader(waveforms, spike_samples)
        self.all_waveforms = waveforms

        self.template_masks = get_masks(templates)
        self.all_masks = MaskLoader(self.template_masks, self.spike_templates)

        # TODO
        self.cluster_groups = {c: None for c in range(n_clusters)}

    def get_cluster_templates(self, cluster_id):
        spike_ids = self.spikes_per_cluster(cluster_id)
        st = self.spike_templates[spike_ids]
        return np.bincount(st, minlength=self.n_templates)

    def get_background_features(self):
        # Disable for now
        pass

    # def get_waveforms_amplitude(self, cluster_id):
    #     # mm = self.get_mean_masks(cluster_id)
    #     # mw = self.get_mean_waveforms(cluster_id)
    #     # TODO: use clusters instead of templates
    #     tmp = self.templates[[cluster_id]]
    #     mm = get_masks(tmp)
    #     assert mw.ndim == 2
    #     return get_waveform_amplitude(mm, mw)

    def _init_context(self):
        super(TemplateController, self)._init_context()
        ctx = self.context
        self.get_amplitudes = concat_per_cluster(
            ctx.cache(self.get_amplitudes))
        self.get_cluster_templates = ctx.cache(self.get_cluster_templates)
        self.get_cluster_pair_features = ctx.cache(
            self.get_cluster_pair_features)

    def get_waveforms(self, cluster_id):
        # Waveforms.
        waveforms_b = self._select_data(cluster_id,
                                        self.all_waveforms,
                                        100,  # TODO
                                        )
        m = waveforms_b.data.mean(axis=1).mean(axis=1)
        waveforms_b.data -= m[:, np.newaxis, np.newaxis]
        # Find the templates corresponding to the cluster.
        template_ids = np.nonzero(self.get_cluster_templates(cluster_id))[0]
        # Templates.
        templates = self.templates_unw[template_ids]
        assert templates.ndim == 3
        masks = self.template_masks[template_ids]
        assert masks.ndim == 2
        assert templates.shape[0] == masks.shape[0]
        # Find mean amplitude.
        spike_ids = waveforms_b.spike_ids
        mean_amp = self.all_amplitudes[spike_ids].mean()
        tmp = templates * mean_amp
        template_b = Bunch(spike_ids=template_ids,
                           spike_clusters=template_ids,
                           data=tmp,
                           masks=masks,
                           alpha=1.,
                           )
        return [waveforms_b, template_b]

    def get_features(self, cluster_id, load_all=False):
        # TODO: load all features
        # Overriden to take into account the sparse structure.
        spike_ids = self._select_spikes(cluster_id, 1000)
        st = self.spike_templates[spike_ids]
        nc = self.n_channels
        nfpc = self.n_features_per_channel
        ns = len(spike_ids)
        f = _densify(spike_ids, self.all_features,
                     self.features_ind[st, :], self.n_channels)
        f = np.transpose(f, (0, 2, 1))
        assert f.shape == (ns, nc, nfpc)
        b = Bunch()
        b.data = f
        b.spike_ids = spike_ids
        b.spike_clusters = self.spike_clusters[spike_ids]
        b.masks = self.all_masks[spike_ids]
        return b

    def get_amplitudes(self, cluster_id):
        spike_ids = self._select_spikes(cluster_id, 10000)
        d = Bunch()
        d.spike_ids = spike_ids
        d.spike_clusters = cluster_id * np.ones(len(spike_ids), dtype=np.int32)
        d.x = self.spike_times[spike_ids]
        d.y = self.all_amplitudes[spike_ids]
        return d

    def _get_template_features(self, spike_ids):
        tf = self.template_features
        tfi = self.template_features_ind
        # For each spike, the non-zero columns.
        ind = tfi[self.spike_templates[spike_ids]]
        return _densify(spike_ids, tf, ind, self.n_templates)

    def get_cluster_pair_features(self, ci, cj):
        si = self._select_spikes(ci)
        sj = self._select_spikes(cj)

        ni = self.get_cluster_templates(ci)
        nj = self.get_cluster_templates(cj)

        ti = self._get_template_features(si)
        x0 = np.sum(ti * ni[np.newaxis, :], axis=1) / ni.sum()
        y0 = np.sum(ti * nj[np.newaxis, :], axis=1) / nj.sum()

        tj = self._get_template_features(sj)
        x1 = np.sum(tj * ni[np.newaxis, :], axis=1) / ni.sum()
        y1 = np.sum(tj * nj[np.newaxis, :], axis=1) / nj.sum()

        d = Bunch()
        d.x = np.hstack((x0, x1))
        d.y = np.hstack((y0, y1))
        d.spike_ids = np.hstack((si, sj))
        d.spike_clusters = self.spike_clusters[d.spike_ids]
        return d

    def get_cluster_features(self, cluster_ids):
        if len(cluster_ids) < 2:
            return None
        cx, cy = map(int, cluster_ids[:2])
        return self.get_cluster_pair_features(cx, cy)

    def get_traces(self, interval):
        """Load traces and spikes in an interval."""
        tr = select_traces(self.all_traces, interval,
                           sample_rate=self.sample_rate,
                           )
        tr = tr - np.mean(tr, axis=0)

        a, b = self.spike_times.searchsorted(interval)
        sc = self.spike_templates[a:b]

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
        sim = []
        sim0 = []
        if self.template_features_ind is not None:
            # Find the templates corresponding to the cluster.
            count = self.get_cluster_templates(cluster_id)
            sim_templates = np.argsort(count)[::-1]
            # Only keep templates corresponding to the cluster.
            n = np.sum(count > 0)
            sim_templates = sim_templates[:n]
            # Sort them by decreasing size.
            # Add the similar templates.
            sim0 = []
            for tid in sim_templates:
                sim0.extend([tmp for tmp in self.template_features_ind[tid]
                             if tmp not in sim0])
            n = len(sim0)
            sim.extend([(int(c), -n + i) for i, c in enumerate(sim0)])

        sim1 = self.get_close_clusters(cluster_id)
        sim1 = [_ for _ in sim1 if _[0] not in sim0]
        sim.extend(sim1)
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
        view = FeatureTemplateView(coords=self.get_cluster_features,
                                   )
        view.attach(gui)
        return view


#------------------------------------------------------------------------------
# Template GUI
#------------------------------------------------------------------------------

def create_template_gui(dat_path=None, plugins=None, **kwargs):
    controller = TemplateController(dat_path, **kwargs)
    # Create the GUI.
    gui = create_gui(name='TemplateGUI',
                     subtitle=dat_path,
                     plugins=plugins,
                     )
    controller.set_manual_clustering(gui)
    controller.add_waveform_view(gui)

    controller.add_amplitude_view(gui)
    controller.add_trace_view(gui)
    controller.add_correlogram_view(gui)

    if controller.all_features is not None:
        controller.add_feature_view(gui)
    if controller.template_features is not None:
        controller.add_feature_template_view(gui)

    # Save.
    @gui.connect_
    def on_request_save(spike_clusters, groups):
        # TODO
        pass

    return gui
