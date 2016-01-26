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
                          get_closest_clusters,
                          Selector,
                          )
from phy.io.context import Context
from phy.stats.clusters import (mean,
                                get_max_waveform_amplitude,
                                get_mean_masked_features_distance,
                                get_unmasked_channels,
                                get_sorted_main_channels,
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

    # Model data.
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
    def feature_lim():
        """Return the max of a subset of the feature amplitudes."""
        return _get_data_lim(model.all_features,
                             max_n_spikes_per_cluster['feature_lim'])
    model.feature_lim = feature_lim

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

    @context.cache
    def waveform_lim():
        """Return the max of a subset of the waveform amplitudes."""
        return _get_data_lim(model.all_waveforms,
                             max_n_spikes_per_cluster['waveform_lim'])
    model.waveform_lim = waveform_lim

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

    def traces(interval):
        """Load traces and spikes in an interval."""
        tr = select_traces(model.all_traces, interval,
                           sample_rate=model.sample_rate,
                           ).copy()
        return [tr]
    model.traces = traces

    def spikes_traces(interval):
        traces = model.traces(interval)[0]
        b = extract_spikes(traces, interval,
                           sample_rate=model.sample_rate,
                           spike_times=model.spike_times,
                           spike_clusters=model.spike_clusters,
                           all_masks=model.all_masks,
                           n_samples_waveforms=model.n_samples_waveforms,
                           )
        return b
    model.spikes_traces = spikes_traces

    # Mean quantities.
    # -------------------------------------------------------------------------

    @context.cache
    def mean_masks(cluster_id):
        # We access [1] because we return spike_ids, masks.
        return mean(model.masks(cluster_id).masks)
    model.mean_masks = mean_masks

    @context.cache
    def mean_features(cluster_id):
        return mean(model.features(cluster_id).features)
    model.mean_features = mean_features

    @context.cache
    def mean_waveforms(cluster_id):
        return mean(model.waveforms(cluster_id).waveforms)
    model.mean_waveforms = mean_waveforms

    # Statistics.
    # -------------------------------------------------------------------------

    @context.cache(memcache=True)
    def best_channels(cluster_id):
        mm = model.mean_masks(cluster_id)
        uch = get_unmasked_channels(mm)
        return get_sorted_main_channels(mm, uch)
    model.best_channels = best_channels

    @context.cache(memcache=True)
    def best_channels_multiple(cluster_ids):
        best_channels = []
        for cluster in cluster_ids:
            channels = model.best_channels(cluster)
            best_channels.extend([ch for ch in channels
                                  if ch not in best_channels])
        return best_channels
    model.best_channels_multiple = best_channels_multiple

    @context.cache(memcache=True)
    def max_waveform_amplitude(cluster_id):
        mm = model.mean_masks(cluster_id)
        mw = model.mean_waveforms(cluster_id)
        assert mw.ndim == 2
        return np.asscalar(get_max_waveform_amplitude(mm, mw))
    model.max_waveform_amplitude = max_waveform_amplitude

    @context.cache(memcache=None)
    def mean_masked_features_score(cluster_0, cluster_1):
        mf0 = model.mean_features(cluster_0)
        mf1 = model.mean_features(cluster_1)
        mm0 = model.mean_masks(cluster_0)
        mm1 = model.mean_masks(cluster_1)
        nfpc = model.n_features_per_channel
        d = get_mean_masked_features_distance(mf0, mf1, mm0, mm1,
                                              n_features_per_channel=nfpc)
        s = 1. / max(1e-10, d)
        return s
    model.mean_masked_features_score = mean_masked_features_score

    @context.cache(memcache=True)
    def most_similar_clusters(cluster_id):
        assert isinstance(cluster_id, int)
        return get_closest_clusters(cluster_id, model.cluster_ids,
                                    model.mean_masked_features_score,
                                    max_n_similar_clusters)
    model.most_similar_clusters = most_similar_clusters

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
    @context.cache(memcache=True)
    def spikes_per_cluster(cluster_id):
        return np.nonzero(model.spike_clusters == cluster_id)[0]
    model.spikes_per_cluster = spikes_per_cluster

    selector = Selector(model.spikes_per_cluster)
    create_cluster_store(model, selector=selector, context=context)

    return model
