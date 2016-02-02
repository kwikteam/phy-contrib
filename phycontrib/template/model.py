# -*- coding: utf-8 -*-

"""Template matching self."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import os.path as op

import numpy as np
import scipy.io as sio

from phy.traces import SpikeLoader, WaveformLoader
from phy.traces.filter import apply_filter, bandpass_filter

from phycontrib.kwik.model import _concatenate_virtual_arrays
from phycontrib.csicsvari.traces import read_dat


filenames = {
    'amplitudes': 'amplitudes.npy',
    'spike_clusters': 'clusterIDs.npy',
    'templates': 'templates.npy',
    'spike_samples': 'spikeTimes.npy',
    'channel_mapping': 'chanMap0ind.npy',
    'channel_positions_x': 'xcoords.npy',
    'channel_positions_y': 'ycoords.npy',
    'whitening_matrix': 'whiteningMatrix.npy',

    'features': 'pcFeatures.npy',
    'features_ind': 'pcFeatureInds.npy',
    'template_features': 'templateFeatures.npy',
    'template_features_ind': 'templateFeatureInds.npy',
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
    def __init__(self, cluster_masks, spike_clusters):
        self._spike_clusters = spike_clusters
        self._cluster_masks = cluster_masks
        self.shape = (len(spike_clusters), cluster_masks.shape[1])

    def __getitem__(self, item):
        # item contains spike ids
        clu = self._spike_clusters[item]
        return self._cluster_masks[clu]


class TemplateModel(object):
    def __init__(self, dat_path=None,
                 n_channels_dat=None,
                 dtype=None,
                 sample_rate=None,
                 n_samples_waveforms=None,
                 ):

        traces = read_dat(dat_path,
                          n_channels=n_channels_dat,
                          dtype=dtype or np.int16,
                          )

        n_samples_t, _ = traces.shape
        assert _ == n_channels_dat

        amplitudes = read_array('amplitudes').squeeze()
        n_spikes, = amplitudes.shape

        spike_clusters = read_array('spike_clusters').squeeze()
        spike_clusters = spike_clusters.astype(np.int32)
        assert spike_clusters.shape == (n_spikes,)

        spike_samples = read_array('spike_samples').squeeze()
        assert spike_samples.shape == (n_spikes,)

        templates = read_array('templates')
        templates[np.isnan(templates)] = 0
        templates = np.transpose(templates, (2, 1, 0))
        n_templates, n_samples_templates, n_channels = templates.shape

        channel_mapping = read_array('channel_mapping').squeeze()
        channel_mapping = channel_mapping.astype(np.int32)
        assert channel_mapping.shape == (n_channels,)

        channel_positions = np.c_[read_array('channel_positions_x'),
                                  read_array('channel_positions_y')]
        assert channel_positions.shape == (n_channels, 2)

        all_features = np.load(filenames['features'], mmap_mode='r')
        features_ind = read_array('features_ind').astype(np.int32)

        self.all_features = all_features
        self.features_ind = features_ind

        template_features = np.load(filenames['template_features'],
                                    mmap_mode='r')
        template_features_ind = read_array('template_features_ind'). \
            astype(np.int32)
        template_features_ind = template_features_ind.T.copy()

        self.n_channels = n_channels
        # Take dead channels into account.
        traces = _concatenate_virtual_arrays([traces], channel_mapping)
        self.n_spikes = n_spikes

        # Amplitudes
        self.all_amplitudes = amplitudes
        self.amplitudes_lim = self.all_amplitudes.max()

        # Templates
        self.templates = templates
        self.n_samples_templates = n_samples_templates
        self.template_lim = np.max(np.abs(self.templates))
        self.n_templates = len(self.templates)

        self.sample_rate = sample_rate
        self.duration = n_samples_t / float(self.sample_rate)

        self.spike_times = spike_samples / float(self.sample_rate)
        assert np.all(np.diff(self.spike_times) >= 0)

        self.spike_clusters = spike_clusters
        self.cluster_ids = np.unique(self.spike_clusters)
        n_clusters = len(self.cluster_ids)
        self.channel_positions = channel_positions
        self.all_traces = traces

        self.whitening_matrix = read_array('whitening_matrix')

        # Filter the waveforms.
        order = 3
        filter_margin = order * 3
        b_filter = bandpass_filter(rate=sample_rate,
                                   low=500.,
                                   high=sample_rate * .475,
                                   order=order)

        def the_filter(x, axis=0):
            return apply_filter(x, b_filter, axis=axis)

        # Fetch waveforms from traces.
        waveforms = WaveformLoader(traces=traces,
                                   n_samples_waveforms=n_samples_waveforms,
                                   filter=the_filter,
                                   filter_margin=filter_margin,
                                   )
        waveforms = SpikeLoader(waveforms, spike_samples)
        self.all_waveforms = waveforms

        self.template_masks = get_masks(templates)
        self.all_masks = MaskLoader(self.template_masks, spike_clusters)

        self.n_features_per_channel = 3
        self.n_samples_waveforms = n_samples_waveforms
        # TODO
        self.cluster_groups = {c: None for c in range(n_clusters)}

        # Check sparse features arrays shapes.
        assert all_features.ndim == 3
        n_loc_chan = all_features.shape[2]
        assert all_features.shape == (self.n_spikes,
                                      self.n_features_per_channel,
                                      n_loc_chan,
                                      )
        assert features_ind.shape == (n_loc_chan, self.n_templates)

        n_sim_tem = template_features.shape[1]
        assert template_features.shape == (n_spikes, n_sim_tem)
        assert template_features_ind.shape == (n_templates, n_sim_tem)
        self.template_features_ind = template_features_ind
        self.template_features = template_features

        # Unwhiten the templates.
        wmi = np.linalg.inv(self.whitening_matrix / 200.)
        self.templates_unw = np.dot(self.templates, wmi)
