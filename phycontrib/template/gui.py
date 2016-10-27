# -*- coding: utf-8 -*-

"""Template GUI."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import csv
import glob
import logging
from operator import itemgetter
import os
import os.path as op
import shutil

import click
import numpy as np
import scipy.io as sio

from phy.cluster.manual.controller import Controller
from phy.cluster.manual.views import (select_traces, ScatterView)
from phy.gui import create_app, run_app  # noqa
from phy.io.array import (concat_per_cluster,
                          _concatenate_virtual_arrays,
                          _index_of,
                          _unique,
                          get_excerpts,
                          )
from phy.plot.transform import _normalize
from phy.utils.cli import _run_cmd
from phy.stats.clusters import get_waveform_amplitude
from phy.traces import WaveformLoader
from phy.utils import Bunch, IPlugin
from phy.utils.cli import _add_log_file
from phy.utils._misc import _read_python

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


class AmplitudeView(ScatterView):
    pass


class FeatureTemplateView(ScatterView):
    pass


#------------------------------------------------------------------------------
# Template Controller
#------------------------------------------------------------------------------

filenames = {
    'spike_templates': 'spike_templates.npy',
    'spike_clusters': 'spike_clusters.npy',
    'cluster_groups': 'cluster_groups.csv',

    'spike_samples': 'spike_times.npy',
    'amplitudes': 'amplitudes.npy',
    'templates': 'templates.npy',
    'templates_unw': 'templates_unw.npy',

    'channel_mapping': 'channel_map.npy',
    'channel_positions': 'channel_positions.npy',
    'whitening_matrix': 'whitening_mat.npy',

    'features': 'pc_features.npy',
    'features_ind': 'pc_feature_ind.npy',
    'features_spike_ids': 'pc_feature_spike_ids.npy',

    'template_features': 'template_features.npy',
    'template_features_ind': 'template_feature_ind.npy',

    'similar_templates': 'similar_templates.npy',
}


def read_array(name):
    fn = filenames[name]
    arr_name, ext = op.splitext(fn)
    if ext == '.mat':
        return sio.loadmat(fn)[arr_name]
    elif ext == '.npy':
        return np.load(fn)


def write_array(name, arr):
    np.save(name, arr)


def get_closest_channels(channel_positions, n=None):
    """Return a (n_channels, n) array with the closest channels to
    each channel.
    """
    x = channel_positions[:, 0]
    y = channel_positions[:, 1]
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    d = dx ** 2 + dy ** 2
    out = np.argsort(d, axis=1)
    if n:
        out = out[:, :n]
    return out


def get_masks(templates, closest_channels):
    # TODO: precompute channels to each channel
    n_templates, n_samples_templates, n_channels = templates.shape
    # Template max.
    amp = templates.max(axis=1)  # (n_templates, n_channels)
    # Template min.
    mi = templates.min(axis=1)  # (n_templates, n_channels)
    # Template amplitudes across all channels.
    amp -= mi  # (n_templates, n_channels)
    # The best channel for each template.
    best = np.argmax(amp, axis=1)  # (n_templates,)
    # Create the masks array.
    masks = np.zeros((n_templates, n_channels))
    assert closest_channels.shape[0] == n_channels
    n = closest_channels.shape[1]
    # For each template, the closest channels.
    ch = closest_channels[best, :]  # (n_templates, n)
    x = np.arange(n_templates)
    x = np.tile(x[:, np.newaxis], (1, n))
    masks[x, ch] = 1
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


def load_metadata(filename, cluster_ids):
    dic = {cluster_id: None for cluster_id in cluster_ids}
    if not op.exists(filename):
        return dic
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        # Skip the header.
        for row in reader:
            break
        for row in reader:
            cluster, value = row
            cluster = int(cluster)
            dic[cluster] = value
    return dic


def save_metadata(filename, field_name, metadata):
    """Save metadata in a CSV file."""
    import sys
    if sys.version_info[0] < 3:
        file = open(filename, 'wb')
    else:
        file = open(filename, 'w', newline='')
    with file as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['cluster_id', field_name])
        writer.writerows([(cluster, metadata[cluster])
                          for cluster in sorted(metadata)])


class TemplateController(Controller):
    gui_name = 'TemplateGUI'

    def __init__(self, dat_path=None, **kwargs):
        dat_path = dat_path or ''
        dir_path = (op.dirname(op.realpath(op.expanduser(dat_path)))
                    if dat_path else os.getcwd())
        self.cache_dir = op.join(dir_path, '.phy')
        self.dat_path = dat_path
        self.__dict__.update(kwargs)
        super(TemplateController, self).__init__()

    def _init_data(self):
        if op.exists(self.dat_path):
            logger.debug("Loading traces at `%s`.", self.dat_path)
            traces = _dat_to_traces(self.dat_path,
                                    n_channels=self.n_channels_dat,
                                    dtype=self.dtype or np.int16,
                                    offset=self.offset,
                                    )
            n_samples_t, _ = traces.shape
            assert _ == self.n_channels_dat
        else:
            if self.dat_path is not None:
                logger.warning("Error while loading data: File %s not found.",
                               self.dat_path)
            traces = None
            n_samples_t = 0

        logger.debug("Loading amplitudes.")
        amplitudes = read_array('amplitudes').squeeze()
        n_spikes, = amplitudes.shape
        self.n_spikes = n_spikes

        # Create spike_clusters if the file doesn't exist.
        if not op.exists(filenames['spike_clusters']):
            shutil.copy(filenames['spike_templates'],
                        filenames['spike_clusters'])
        logger.debug("Loading %d spike clusters.", self.n_spikes)
        spike_clusters = read_array('spike_clusters').squeeze()
        spike_clusters = spike_clusters.astype(np.int32)
        assert spike_clusters.shape == (n_spikes,)
        self.spike_clusters = spike_clusters

        logger.debug("Loading spike templates.")
        spike_templates = read_array('spike_templates').squeeze()
        spike_templates = spike_templates.astype(np.int32)
        assert spike_templates.shape == (n_spikes,)
        self.spike_templates = spike_templates

        logger.debug("Loading spike samples.")
        spike_samples = read_array('spike_samples').squeeze()
        assert spike_samples.shape == (n_spikes,)

        logger.debug("Loading templates.")
        templates = read_array('templates')
        templates[np.isnan(templates)] = 0
        # templates = np.transpose(templates, (2, 1, 0))

        # Unwhiten the templates.
        logger.debug("Loading the whitening matrix.")
        self.whitening_matrix = read_array('whitening_matrix')

        if op.exists(filenames['templates_unw']):
            logger.debug("Loading unwhitened templates.")
            templates_unw = read_array('templates_unw')
            templates_unw[np.isnan(templates_unw)] = 0
        else:
            logger.debug("Couldn't find unwhitened templates, computing them.")
            logger.debug("Inversing the whitening matrix %s.",
                         self.whitening_matrix.shape)
            wmi = np.linalg.inv(self.whitening_matrix)
            logger.debug("Unwhitening the templates %s.",
                         templates.shape)
            templates_unw = np.dot(np.ascontiguousarray(templates),
                                   np.ascontiguousarray(wmi))
            # Save the unwhitened templates.
            write_array('templates_unw.npy', templates_unw)

        n_templates, n_samples_templates, n_channels = templates.shape
        self.n_templates = n_templates

        logger.debug("Loading similar templates.")
        self.similar_templates = read_array('similar_templates')
        assert self.similar_templates.shape == (self.n_templates,
                                                self.n_templates)

        logger.debug("Loading channel mapping.")
        channel_mapping = read_array('channel_mapping').squeeze()
        channel_mapping = channel_mapping.astype(np.int32)
        assert channel_mapping.shape == (n_channels,)
        # Ensure that the mappings maps to valid columns in the dat file.
        assert np.all(channel_mapping <= self.n_channels_dat - 1)
        self.channel_order = channel_mapping

        logger.debug("Loading channel positions.")
        channel_positions = read_array('channel_positions')
        assert channel_positions.shape == (n_channels, 2)

        if op.exists(filenames['features']):
            logger.debug("Loading features.")
            all_features = np.load(filenames['features'], mmap_mode='r')
            features_ind = read_array('features_ind').astype(np.int32)
            # Feature subset.
            if op.exists(filenames['features_spike_ids']):
                features_spike_ids = read_array('features_spike_ids') \
                    .astype(np.int32)
                assert len(features_spike_ids) == len(all_features)
                self.features_spike_ids = features_spike_ids
                ns = len(features_spike_ids)
            else:
                ns = self.n_spikes
                self.features_spike_ids = None

            assert all_features.ndim == 3
            n_loc_chan = all_features.shape[2]
            self.n_features_per_channel = all_features.shape[1]
            assert all_features.shape == (ns,
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

        if op.exists(filenames['template_features']):
            logger.debug("Loading template features.")
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
        if traces is not None:
            # Find the scaling factor for the traces.
            scaling = 1. / self._data_lim(traces[:10000])
            traces = _concatenate_virtual_arrays([traces],
                                                 channel_mapping,
                                                 scaling=scaling,
                                                 )
        else:
            scaling = 1.

        # Amplitudes
        self.all_amplitudes = amplitudes
        self.amplitudes_lim = np.max(self.all_amplitudes)

        # Templates
        self.templates = templates
        self.templates_unw = templates_unw
        assert self.templates.shape == self.templates_unw.shape
        self.n_samples_templates = n_samples_templates
        self.n_samples_waveforms = n_samples_templates
        self.template_lim = np.max(np.abs(self.templates))

        self.duration = n_samples_t / float(self.sample_rate)

        self.spike_times = spike_samples / float(self.sample_rate)
        assert np.all(np.diff(self.spike_times) >= 0)

        self.cluster_ids = _unique(self.spike_clusters)
        # n_clusters = len(self.cluster_ids)

        self.channel_positions = channel_positions
        self.all_traces = traces

        # Only filter the data for the waveforms if the traces
        # are not already filtered.
        if not getattr(self, 'hp_filtered', False):
            logger.debug("HP filtering the data for waveforms")
            filter_order = 3
        else:
            filter_order = None

        n_closest_channels = getattr(self, 'max_n_unmasked_channels', 16)
        mask_threshold = getattr(self, 'waveform_mask_threshold', None)
        self.closest_channels = get_closest_channels(self.channel_positions,
                                                     n_closest_channels,
                                                     )
        self.template_masks = get_masks(self.templates, self.closest_channels)
        self.all_masks = MaskLoader(self.template_masks, self.spike_templates)

        # Fetch waveforms from traces.
        nsw = self.n_samples_waveforms
        if traces is not None:
            waveforms = WaveformLoader(traces=traces,
                                       masks=self.all_masks,
                                       spike_samples=spike_samples,
                                       n_samples_waveforms=nsw,
                                       filter_order=filter_order,
                                       sample_rate=self.sample_rate,
                                       mask_threshold=mask_threshold,
                                       )
        else:
            waveforms = None
        self.all_waveforms = waveforms

        # Read the cluster groups.
        logger.debug("Loading the cluster groups.")
        self.cluster_groups = load_metadata(filenames['cluster_groups'],
                                            self.cluster_ids)

    def _set_manual_clustering(self):
        super(TemplateController, self)._set_manual_clustering()
        mc = self.manual_clustering
        # Load labels.
        files = glob.glob('*.csv')
        for filename in files:
            if filename == 'cluster_groups.csv':
                continue
            field_name = op.basename(filename)
            values = load_metadata(filename, self.cluster_ids)
            for cluster_id in self.cluster_ids:
                mc.cluster_meta.set(field_name, [cluster_id],
                                    values.get(cluster_id, None),
                                    add_to_stack=False)

    def get_cluster_templates(self, cluster_id):
        spike_ids = self.spikes_per_cluster(cluster_id)
        st = self.spike_templates[spike_ids]
        return np.bincount(st, minlength=self.n_templates)

    def get_channel_noise(self):
        ex = get_excerpts(self.all_traces, n_excerpts=100, excerpt_size=100)
        return ex.std(axis=0)

    def get_background_features(self):
        # Disable for now
        pass

    def get_waveforms_amplitude(self, cluster_id):
        nt = self.get_cluster_templates(cluster_id)
        # Find the template with the highest number of spikes belonging
        # to the selected cluster.
        template_id = np.argmax(nt)
        # Find the masked template waveform amplitude.
        mw = self.templates[[template_id]]
        mm = self.template_masks[[template_id]]
        mw = mw[0, ...]
        mm = mm[0, ...]
        assert mw.ndim == 2
        assert mw.ndim == 2
        return get_waveform_amplitude(mm, mw)

    def _init_context(self):
        super(TemplateController, self)._init_context()
        ctx = self.context
        self.get_amplitudes = concat_per_cluster(
            ctx.memcache(self.get_amplitudes))
        self.get_cluster_templates = ctx.memcache(self.get_cluster_templates)
        self.get_cluster_pair_features = ctx.memcache(
            self.get_cluster_pair_features)
        self._cluster_template_similarities = ctx.memcache(
            self._cluster_template_similarities)
        self._sim_ij = ctx.memcache(self._sim_ij)

        self.get_channel_noise = ctx.cache(self.get_channel_noise)

    def get_waveform_lims(self):
        n_spikes = self.n_spikes_waveforms_lim
        arr = self.all_waveforms
        if arr is None:
            return 0., 1.
        n = arr.shape[0]
        k = max(1, n // n_spikes)
        # Extract waveforms.
        arr = arr[::k].astype(np.float64)
        mean = arr.mean(axis=1).mean(axis=1)
        arr -= mean[:, np.newaxis, np.newaxis]
        # Take the corresponding masks.
        masks = self.all_masks[::k].copy()
        arr = arr * masks[:, np.newaxis, :]
        # NOTE: on some datasets, there are a few outliers that screw up
        # the normalization. These parameters should be customizable.
        m = np.percentile(arr, .05)
        M = np.percentile(arr, 99.95)
        return m, M

    def get_waveforms(self, cluster_id):
        m, M = self.get_waveform_lims()
        if self.all_waveforms is not None:
            # Waveforms.
            waveforms_b = self._select_data(cluster_id,
                                            self.all_waveforms,
                                            self.n_spikes_waveforms,
                                            )
            w = waveforms_b.data
            # Sparsify.
            channels = np.nonzero(w.mean(axis=1).mean(axis=0))[0]
            w = w[:, :, channels]
            waveforms_b.channels = channels
            # Normalize.
            mean = w.mean(axis=1).mean(axis=1)
            w = w.astype(np.float64)
            w -= mean[:, np.newaxis, np.newaxis]
            w = _normalize(w, m, M)
            waveforms_b.data = w
            waveforms_b.cluster_id = cluster_id
            waveforms_b.tag = 'waveforms'
        else:
            waveforms_b = None
        # Find the templates corresponding to the cluster.
        template_ids = np.nonzero(self.get_cluster_templates(cluster_id))[0]
        # Templates.
        templates = self.templates_unw[template_ids]
        assert templates.ndim == 3
        # Masks.
        masks = self.template_masks[template_ids]
        assert masks.ndim == 2
        assert templates.shape[0] == masks.shape[0]
        # Find mean amplitude.
        spike_ids = self._select_spikes(cluster_id,
                                        self.n_spikes_waveforms_lim)
        mean_amp = self.all_amplitudes[spike_ids].mean()
        # Normalize.
        # mean = templates.mean(axis=1).mean(axis=1)
        templates = templates.astype(np.float64).copy()
        # templates -= mean[:, np.newaxis, np.newaxis]
        templates *= mean_amp
        templates *= 2. / (M - m)
        template_b = Bunch(data=templates,
                           masks=masks,
                           alpha=1.,
                           cluster_id=cluster_id,
                           tag='templates',
                           )
        if waveforms_b is not None:
            return [waveforms_b, template_b]
        else:
            return [template_b]

    def get_features(self, cluster_id, load_all=False):
        # Overriden to take into account the sparse structure.
        # Only keep spikes belonging to the features spike ids.
        if self.features_spike_ids is not None:
            # All spikes
            spike_ids = self._select_spikes(cluster_id)
            spike_ids = np.intersect1d(spike_ids, self.features_spike_ids)
            # Relative indices of the spikes in the self.features_spike_ids
            # array, necessary to load features from all_features which only
            # contains the subset of the spikes.
            spike_ids_rel = _index_of(spike_ids, self.features_spike_ids)
        else:
            spike_ids = self._select_spikes(cluster_id,
                                            self.n_spikes_features
                                            if not load_all else None)
            spike_ids_rel = spike_ids
        st = self.spike_templates[spike_ids]
        nc = self.n_channels
        nfpc = self.n_features_per_channel
        ns = len(spike_ids)
        f = _densify(spike_ids_rel, self.all_features,
                     self.features_ind[st, :], self.n_channels)
        f = np.transpose(f, (0, 2, 1))
        assert f.shape == (ns, nc, nfpc)
        b = Bunch()

        # Normalize features.
        m = self.get_feature_lim()
        f = _normalize(f, -m, m)

        b.data = f
        b.spike_ids = spike_ids
        b.spike_clusters = self.spike_clusters[spike_ids]
        b.masks = self.all_masks[spike_ids]
        return b

    def get_amplitudes(self, cluster_id):
        spike_ids = self._select_spikes(cluster_id, self.n_spikes_features)
        d = Bunch()
        d.spike_ids = spike_ids
        d.x = self.spike_times[spike_ids]
        d.y = self.all_amplitudes[spike_ids]
        M = d.y.max()
        d.data_bounds = [0, 0, self.duration, M]
        return d

    def _get_template_features(self, spike_ids):
        tf = self.template_features
        tfi = self.template_features_ind
        # For each spike, the non-zero columns.
        ind = tfi[self.spike_templates[spike_ids]]
        return _densify(spike_ids, tf, ind, self.n_templates)

    def get_cluster_pair_features(self, ci, cj):
        si = self._select_spikes(ci, self.n_spikes_features)
        sj = self._select_spikes(cj, self.n_spikes_features)

        ni = self.get_cluster_templates(ci)
        nj = self.get_cluster_templates(cj)

        ti = self._get_template_features(si)
        x0 = np.average(ti, weights=ni, axis=1)
        y0 = np.average(ti, weights=nj, axis=1)

        tj = self._get_template_features(sj)
        x1 = np.average(tj, weights=ni, axis=1)
        y1 = np.average(tj, weights=nj, axis=1)

        # Compute the data bounds.
        x_min = min(x0.min(), x1.min())
        y_min = min(y0.min(), y1.min())
        x_max = max(x0.max(), x1.max())
        y_max = max(y0.max(), y1.max())
        data_bounds = (x_min, y_min, x_max, y_max)

        return [Bunch(x=x0, y=y0, spike_ids=si, data_bounds=data_bounds),
                Bunch(x=x1, y=y1, spike_ids=sj, data_bounds=data_bounds)]

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
                Bunch(traces=tr_sub, color=(.5, .5, .5, .75)),
                ]

    def _cluster_template_similarities(self, cluster_id):
        # Templates of the cluster.
        temp = np.nonzero(self.get_cluster_templates(cluster_id))[0]
        # Max similarity of cluster_id with all templates.
        return np.max(self.similar_templates[temp, :], axis=0)

    def _sim_ij(self, ci, cj):
        """Similarity between two clusters."""
        # The similarity of the cluster with each template.
        sims = self._cluster_template_similarities(ci)
        # Templates of the cluster.
        if cj < self.n_templates:
            return sims[cj]
        temp = np.nonzero(self.get_cluster_templates(cj))[0]
        return np.max(sims[temp])

    def similarity(self, cluster_id):
        out = [(cj, self._sim_ij(cluster_id, cj)) for cj in self.cluster_ids]
        return sorted(out, key=itemgetter(1), reverse=True)

    def add_amplitude_view(self, gui):
        v = AmplitudeView(coords=self.get_amplitudes)
        return self._add_view(gui, v)

    def add_feature_template_view(self, gui):
        v = FeatureTemplateView(coords=self.get_cluster_features)
        return self._add_view(gui, v)

    def create_gui(self, config_dir=None):
        """Create the template GUI."""
        f = super(TemplateController, self).create_gui
        gui = f(name=self.gui_name,
                subtitle=self.dat_path,
                config_dir=config_dir,
                )

        # Add custom views for the template GUI.
        if self.all_amplitudes is not None:
            self.add_amplitude_view(gui)
        if self.template_features is not None:
            self.add_feature_template_view(gui)

        # Add the waveform view even if there is no raw data.
        if self.all_waveforms is None:
            self.add_waveform_view(gui)

        # Add the option to show/hide waveforms.
        waveform_view = gui.get_view('WaveformView', is_visible=False)
        if waveform_view:
            @waveform_view.actions.add(shortcut='w')
            def toggle_waveforms():
                """Show or hide the waveforms in the waveform view."""
                if not waveform_view.filtered_tags:
                    waveform_view.filter_by_tag('templates')
                else:
                    waveform_view.filter_by_tag()

        # Save.
        @gui.connect_
        def on_request_save(spike_clusters, groups, *labels):
            # Save the clusters.
            np.save(filenames['spike_clusters'], spike_clusters)
            # Save cluster groups.
            save_metadata(filenames['cluster_groups'], 'group', groups)
            # Save other labels.
            for field_name, dic in labels:
                save_metadata('cluster_%s.csv' % field_name, field_name, dic)

        # Save the memcache when closing the GUI.
        @gui.connect_
        def on_close():
            self.context.save_memcache()

        # Add split on templates action.
        mc = self.manual_clustering

        @mc.actions.add(shortcut='shift+ctrl+k')
        def split_init(cluster_ids=None):
            """Split a cluster according to the original templates."""
            if cluster_ids is None:
                cluster_ids = mc.selected
            spike_ids = mc.clustering.spikes_in_clusters(cluster_ids)
            mc.split(spike_ids, self.spike_templates[spike_ids])

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
