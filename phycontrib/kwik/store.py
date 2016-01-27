# -*- coding: utf-8 -*-

"""Cluster store."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging
import os.path as op
import shutil

import numpy as np

from phy.io.array import (concat_per_cluster,
                          _get_data_lim,
                          Selector,
                          )
from phy.io.context import Context
from phy.stats.clusters import (mean,
                                get_waveform_amplitude,
                                )
from phy.cluster.manual.views import select_traces, extract_spikes
from phy.utils import Bunch
from .model import KwikModel

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Cluster store
# -----------------------------------------------------------------------------

def create_cluster_store(model, selector=None, context=None):
    assert model
    assert context

    # TODO: make this configurable.
    max_n_spikes_per_cluster = {
        'masks': 1000,
        'features': 1000,
        'background_features': 1000,
        'waveforms': 100,
        'waveform_lim': 1000,  # used to compute the waveform bounds
        'feature_lim': 1000,  # used to compute the waveform bounds
    }
    max_n_similar_clusters = 20

    def select(cluster_id, n=None):
        assert isinstance(cluster_id, int)
        assert cluster_id >= 0
        return selector.select_spikes([cluster_id], max_n_spikes_per_cluster=n)

    def _get_data(**kwargs):
        kwargs['spike_clusters'] = model.spike_clusters[kwargs['spike_ids']]
        return Bunch(**kwargs)

    # Masks
    # -------------------------------------------------------------------------

    @concat_per_cluster
    @context.cache
    def masks(cluster_id):
        spike_ids = select(cluster_id, max_n_spikes_per_cluster['masks'])
        if model.all_masks is None:
            masks = np.ones((len(spike_ids), model.n_channels))
        else:
            masks = np.atleast_2d(model.all_masks[spike_ids])
        assert masks.ndim == 2
        return _get_data(spike_ids=spike_ids,
                         masks=masks,
                         )
    model.masks = masks

    # Features
    # -------------------------------------------------------------------------

    @concat_per_cluster
    @context.cache
    def features(cluster_id):
        spike_ids = select(cluster_id, max_n_spikes_per_cluster['features'])
        fm = np.atleast_3d(model.all_features_masks[spike_ids])
        ns = fm.shape[0]
        nc = model.n_channels
        nfpc = model.n_features_per_channel
        assert fm.ndim == 3
        f = fm[..., 0].reshape((ns, nc, nfpc))
        m = fm[:, ::nfpc, 1]
        return _get_data(spike_ids=spike_ids,
                         features=f,
                         masks=m,
                         )
    model.features = features

    @context.cache
    def background_features():
        n = max_n_spikes_per_cluster['background_features']
        k = max(1, model.n_spikes // n)
        features = model.all_features[::k]
        masks = model.all_masks[::k]
        spike_ids = np.arange(0, model.n_spikes, k)
        assert spike_ids.shape == (features.shape[0],)
        assert features.ndim == 3
        assert masks.ndim == 2
        assert masks.shape[0] == features.shape[0]
        return _get_data(spike_ids=spike_ids,
                         features=features,
                         masks=masks,
                         )
    model.background_features = background_features

    @context.memcache
    @context.cache
    def feature_lim():
        """Return the max of a subset of the feature amplitudes."""
        return _get_data_lim(model.all_features,
                             max_n_spikes_per_cluster['feature_lim'])
    model.feature_lim = feature_lim

    # Waveforms
    # -------------------------------------------------------------------------

    @concat_per_cluster
    @context.cache
    def waveforms(cluster_id):
        spike_ids = select(cluster_id,
                           max_n_spikes_per_cluster['waveforms'])
        waveforms = np.atleast_2d(model.all_waveforms[spike_ids])
        assert waveforms.ndim == 3
        masks = np.atleast_2d(model.all_masks[spike_ids])
        assert masks.ndim == 2
        # Ensure that both arrays have the same number of channels.
        assert masks.shape[1] == waveforms.shape[2]
        return _get_data(spike_ids=spike_ids,
                         waveforms=waveforms,
                         masks=masks,
                         )
    model.waveforms = waveforms

    @context.memcache
    @context.cache
    def waveform_lim():
        """Return the max of a subset of the waveform amplitudes."""
        return _get_data_lim(model.all_waveforms,
                             max_n_spikes_per_cluster['waveform_lim'])
    model.waveform_lim = waveform_lim

    # Traces
    # -------------------------------------------------------------------------

    def traces(interval):
        """Load traces and spikes in an interval."""
        tr = select_traces(model.all_traces, interval,
                           sample_rate=model.sample_rate,
                           )
        tr = tr - np.mean(tr, axis=0)
        return [tr]
    model.traces = traces

    def spikes_traces(interval, traces):
        traces = traces[0]
        return extract_spikes(traces, interval,
                              sample_rate=model.sample_rate,
                              spike_times=model.spike_times,
                              spike_clusters=model.spike_clusters,
                              all_masks=model.all_masks,
                              n_samples_waveforms=model.n_samples_waveforms,
                              )
    model.spikes_traces = spikes_traces

    # Mean quantities.
    # -------------------------------------------------------------------------

    @context.memcache
    def mean_masks(cluster_id):
        # We access [1] because we return spike_ids, masks.
        return mean(model.masks(cluster_id).masks)
    model.mean_masks = mean_masks

    @context.memcache
    def mean_features(cluster_id):
        return mean(model.features(cluster_id).features)
    model.mean_features = mean_features

    @context.memcache
    def mean_waveforms(cluster_id):
        return mean(model.waveforms(cluster_id).waveforms)
    model.mean_waveforms = mean_waveforms

    # Statistics.
    # -------------------------------------------------------------------------

    @context.memcache
    def waveforms_amplitude(cluster_id):
        mm = model.mean_masks(cluster_id)
        mw = model.mean_waveforms(cluster_id)
        assert mw.ndim == 2
        return get_waveform_amplitude(mm, mw)
    model.waveforms_amplitude = waveforms_amplitude

    @context.memcache
    def best_channel(cluster_id):
        wa = model.waveforms_amplitude(cluster_id)
        return int(wa.argmax())
    model.best_channel = best_channel

    @context.memcache
    def best_channel_position(cluster_id):
        cha = model.best_channel(cluster_id)
        return tuple(model.channel_positions[cha])
    model.best_channel_position = best_channel_position

    @context.memcache
    @context.cache
    def closest_clusters(cluster_id):
        assert isinstance(cluster_id, int)
        # Position of the cluster's best channel.
        pos0 = model.best_channel_position(cluster_id)
        n = len(pos0)
        assert n in (2, 3)
        # Positions of all clusters' best channels.
        pos = np.vstack([model.best_channel_position(int(clu))
                         for clu in model.cluster_ids])
        # Distance of all clusters to the current cluster.
        dist = np.sum((pos - pos0) ** 2, axis=1)
        # Closest clusters.
        ind = np.argsort(dist)[:max_n_similar_clusters]
        return [(int(model.cluster_ids[i]), float(dist[i])) for i in ind]
    model.closest_clusters = closest_clusters

    return model


# -----------------------------------------------------------------------------
# Model with cache
# -----------------------------------------------------------------------------

def _backup(path):
    """Backup a file."""
    path_backup = path + '.bak'
    if not op.exists(path_backup):
        logger.info("Backup `%s`.".format(path_backup))
        shutil.copy(path, path_backup)


def create_model(path):
    """Create a model from a .kwik file."""

    # Make a backup of the Kwik file.
    path = op.realpath(op.expanduser(path))
    _backup(path)

    # Open the dataset.
    model = KwikModel(path)

    # Create the context.
    context = Context(op.join(op.dirname(path), '.phy'))

    # Define and cache the cluster -> spikes function.
    @context.memcache
    def spikes_per_cluster(cluster_id):
        return np.nonzero(model.spike_clusters == cluster_id)[0]
    model.spikes_per_cluster = spikes_per_cluster

    selector = Selector(model.spikes_per_cluster)
    create_cluster_store(model, selector=selector, context=context)

    return model
