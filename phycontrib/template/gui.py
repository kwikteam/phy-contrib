# -*- coding: utf-8 -*-

"""Template GUI."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import csv
import logging
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
                          )
from phy.utils.cli import _run_cmd
from phy.stats.clusters import get_waveform_amplitude
from phy.traces import SpikeLoader, WaveformLoader
from phy.traces.filter import apply_filter, bandpass_filter
from phy.utils import Bunch, IPlugin
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
    return np.memmap(dat_path, dtype=dtype, shape=(n_samples, n_channels))


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
    'spike_templates': 'spike_templates.npy',
    'spike_clusters': 'spike_clusters.npy',
    'cluster_groups': 'cluster_groups.csv',

    'spike_samples': 'spike_times.npy',
    'amplitudes': 'amplitudes.npy',
    'templates': 'templates.npy',

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
    gui_name = 'TemplateGUI'

    def __init__(self, dat_path=None, **kwargs):
        path = op.realpath(op.expanduser(dat_path))
        self.cache_dir = op.join(op.dirname(path), '.phy')
        self.dat_path = dat_path
        self.__dict__.update(kwargs)
        super(TemplateController, self).__init__()

    def _init_data(self):
        if op.exists(self.dat_path):
            traces = _dat_to_traces(self.dat_path,
                                    n_channels=self.n_channels_dat,
                                    dtype=self.dtype or np.int16,
                                    offset=self.offset,
                                    )
            n_samples_t, _ = traces.shape
            assert _ == self.n_channels_dat
        else:
            traces = None
            n_samples_t = 0

        amplitudes = read_array('amplitudes').squeeze()
        n_spikes, = amplitudes.shape
        self.n_spikes = n_spikes

        # Create spike_clusters if the file doesn't exist.
        if not op.exists(filenames['spike_clusters']):
            shutil.copy(filenames['spike_templates'],
                        filenames['spike_clusters'])
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

        self.similar_templates = read_array('similar_templates')
        assert self.similar_templates.shape == (self.n_templates,
                                                self.n_templates)

        channel_mapping = read_array('channel_mapping').squeeze()
        channel_mapping = channel_mapping.astype(np.int32)
        assert channel_mapping.shape == (n_channels,)

        channel_positions = read_array('channel_positions')
        assert channel_positions.shape == (n_channels, 2)

        if op.exists(filenames['features']):
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
        wmi = np.linalg.inv(self.whitening_matrix)
        self.templates_unw = np.dot(self.templates, wmi)

        self.duration = n_samples_t / float(self.sample_rate)

        self.spike_times = spike_samples / float(self.sample_rate)
        assert np.all(np.diff(self.spike_times) >= 0)

        self.cluster_ids = np.unique(self.spike_clusters)
        # n_clusters = len(self.cluster_ids)

        self.channel_positions = channel_positions
        self.all_traces = traces

        # Filter the waveforms.
        order = 3
        filter_margin = order * 3
        b_filter = bandpass_filter(rate=self.sample_rate,
                                   low=500.,
                                   high=self.sample_rate * .475,
                                   order=order)

        # Only filter the data for the waveforms if the traces
        # are not already filtered.
        if not getattr(self, 'hp_filtered', False):
            logger.debug("HP filtering the data for waveforms")

            def the_filter(x, axis=0):
                return apply_filter(x, b_filter, axis=axis)
        else:
            the_filter = None

        # Fetch waveforms from traces.
        nsw = self.n_samples_waveforms
        if traces is not None:
            waveforms = WaveformLoader(traces=traces,
                                       n_samples_waveforms=nsw,
                                       filter=the_filter,
                                       filter_margin=filter_margin,
                                       )
            waveforms = SpikeLoader(waveforms, spike_samples)
        else:
            waveforms = None
        self.all_waveforms = waveforms

        self.template_masks = get_masks(templates)
        self.all_masks = MaskLoader(self.template_masks, self.spike_templates)

        # Read the cluster groups.
        self.cluster_groups = {}
        if op.exists(filenames['cluster_groups']):
            with open(filenames['cluster_groups'], 'r') as f:
                reader = csv.reader(f, delimiter='\t')
                # Skip the header.
                for row in reader:
                    break
                for row in reader:
                    cluster, group = row
                    cluster = int(cluster)
                    self.cluster_groups[cluster] = group
        for cluster_id in self.cluster_ids:
            if cluster_id not in self.cluster_groups:
                self.cluster_groups[cluster_id] = None

    def get_cluster_templates(self, cluster_id):
        spike_ids = self.spikes_per_cluster(cluster_id)
        st = self.spike_templates[spike_ids]
        return np.bincount(st, minlength=self.n_templates)

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
        mm = get_masks(mw)
        mw = mw[0, ...]
        mm = mm[0, ...]
        assert mw.ndim == 2
        assert mw.ndim == 2
        return get_waveform_amplitude(mm, mw)

    def _init_context(self):
        super(TemplateController, self)._init_context()
        ctx = self.context
        self.get_amplitudes = concat_per_cluster(
            ctx.cache(self.get_amplitudes))
        self.get_cluster_templates = ctx.cache(self.get_cluster_templates)
        self.get_cluster_pair_features = ctx.cache(
            self.get_cluster_pair_features)

    def get_waveforms(self, cluster_id):
        if self.all_waveforms is not None:
            # Waveforms.
            waveforms_b = self._select_data(cluster_id,
                                            self.all_waveforms,
                                            self.n_spikes_waveforms,
                                            )
            m = waveforms_b.data.mean(axis=1).mean(axis=1)
            waveforms_b.data = waveforms_b.data.astype(np.float64)
            waveforms_b.data -= m[:, np.newaxis, np.newaxis]
        else:
            waveforms_b = None
        # Find the templates corresponding to the cluster.
        template_ids = np.nonzero(self.get_cluster_templates(cluster_id))[0]
        # Templates.
        templates = self.templates_unw[template_ids]
        assert templates.ndim == 3
        masks = self.template_masks[template_ids]
        assert masks.ndim == 2
        assert templates.shape[0] == masks.shape[0]
        # Find mean amplitude.
        spike_ids = self._select_spikes(cluster_id,
                                        self.n_spikes_waveforms_lim)
        mean_amp = self.all_amplitudes[spike_ids].mean()
        tmp = templates * mean_amp
        template_b = Bunch(spike_ids=template_ids,
                           spike_clusters=template_ids,
                           data=tmp,
                           masks=masks,
                           alpha=1.,
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
        b.data = f
        b.spike_ids = spike_ids
        b.spike_clusters = self.spike_clusters[spike_ids]
        b.masks = self.all_masks[spike_ids]
        return b

    def get_amplitudes(self, cluster_id):
        spike_ids = self._select_spikes(cluster_id, self.n_spikes_features)
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
        si = self._select_spikes(ci, self.n_spikes_features)
        sj = self._select_spikes(cj, self.n_spikes_features)

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
                Bunch(traces=tr_sub, color=(.25, .25, .25, .75)),
                ]

    def similarity(self, cluster_id):
        count = self.get_cluster_templates(cluster_id)
        # Load the templates similar to the largest parent template.
        largest_template = np.argmax(count)
        sim = self.similar_templates[largest_template]
        templates = np.argsort(sim)[::-1]
        return [(int(c), sim[c]) for i, c in enumerate(templates)]

    def add_amplitude_view(self, gui):
        v = AmplitudeView(coords=self.get_amplitudes)
        return self._add_view(gui, v)

    def add_feature_template_view(self, gui):
        v = FeatureTemplateView(coords=self.get_cluster_features)
        return self._add_view(gui, v)

    def create_gui(self, plugins=None, config_dir=None):
        """Create the template GUI."""
        create = super(TemplateController, self).create_gui
        gui = create(name=self.gui_name, subtitle=self.dat_path,
                     plugins=plugins, config_dir=config_dir)

        # Add custom views for the template GUI.
        if self.all_amplitudes is not None:
            self.add_amplitude_view(gui)
        if self.template_features is not None:
            self.add_feature_template_view(gui)

        # Save.
        @gui.connect_
        def on_request_save(spike_clusters, groups):
            # Save the clusters.
            np.save(filenames['spike_clusters'], spike_clusters)
            # Save the cluster groups.
            import sys
            if sys.version_info[0] < 3:
                file = open(filenames['cluster_groups'], 'wb')
            else:
                file = open(filenames['cluster_groups'], 'w', newline='')
            with file as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(['cluster_id', 'group'])
                writer.writerows([(cluster, groups[cluster])
                                  for cluster in sorted(groups)])

        # Save the memcache when closing the GUI.
        @gui.connect_
        def on_close():
            self.context.save_memcache()

        return gui


#------------------------------------------------------------------------------
# Template GUI plugin
#------------------------------------------------------------------------------

class TemplateGUIPlugin(IPlugin):
    """Create the `phy template-gui` command for Kwik files."""

    def attach_to_cli(self, cli):

        # Create the `phy cluster-manual file.kwik` command.
        @cli.command('template-gui')
        @click.argument('params-path', type=click.Path(exists=True))
        @click.pass_context
        def cluster_manual(ctx, params_path):
            """Launch the Template GUI on a params.py file."""
            # Create the Qt application.
            create_app()

            params = _read_python(params_path)
            params['dtype'] = np.dtype(params['dtype'])

            controller = TemplateController(**params)
            gui = controller.create_gui()

            gui.show()

            _run_cmd('run_app()', ctx, globals(), locals())

            gui.close()
            del gui
