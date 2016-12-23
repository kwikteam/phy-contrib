# -*- coding: utf-8 -*-

"""Template GUI."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import inspect
import logging
from operator import itemgetter
import os.path as op

import click
import numpy as np

from phy.cluster.supervisor import Supervisor
from phy.cluster.views import (WaveformView,
                               FeatureView,
                               TraceView,
                               CorrelogramView,
                               ScatterView,
                               select_traces,
                               )
from phy.cluster.views.trace import _iter_spike_waveforms
from phy.gui import create_app, run_app, GUI
from phy.io.array import (Selector,
                          )
from phy.io.context import Context, _cache_methods
from phy.stats import correlograms
from phy.utils import Bunch, IPlugin, EventEmitter
from phy.utils._color import ColorSelector
from phy.utils._misc import _read_python
from phy.utils.cli import _run_cmd, _add_log_file

from .model import TemplateModel, from_sparse
from ..utils import attach_plugins

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Utils and views
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


class TemplateFeatureView(ScatterView):
    _callback_delay = 100

    def _get_data(self, cluster_ids):
        if len(cluster_ids) != 2:
            return []
        b = self.coords(cluster_ids)
        return [Bunch(x=b.x0, y=b.y0), Bunch(x=b.x1, y=b.y1)]


class AmplitudeView(ScatterView):
    def _plot_points(self, bunchs, data_bounds):
        super(AmplitudeView, self)._plot_points(bunchs, data_bounds)
        liney = 1.
        self.lines(pos=[[data_bounds[0], liney, data_bounds[2], liney]],
                   data_bounds=data_bounds,
                   color=(1., 1., 1., .5),
                   )


#------------------------------------------------------------------------------
# Template Controller
#------------------------------------------------------------------------------

class TemplateController(EventEmitter):
    gui_name = 'TemplateGUI'

    n_spikes_waveforms = 100
    batch_size_waveforms = 10

    n_spikes_features = 10000
    n_spikes_amplitudes = 10000
    n_spikes_correlograms = 100000

    def __init__(self, dat_path=None, config_dir=None, model=None, **kwargs):
        super(TemplateController, self).__init__()
        if model is None:
            assert dat_path
            dat_path = op.realpath(dat_path)
            self.model = TemplateModel(dat_path, **kwargs)
        else:
            self.model = model
        self.cache_dir = op.join(self.model.dir_path, '.phy')
        self.context = Context(self.cache_dir)
        self.config_dir = config_dir

        self._set_cache()
        self.supervisor = self._set_supervisor()
        self.selector = self._set_selector()
        self.color_selector = ColorSelector()

        self._show_all_spikes = False

        attach_plugins(self, plugins=kwargs.get('plugins', None),
                       config_dir=config_dir)

    # Internal methods
    # -------------------------------------------------------------------------

    def _set_cache(self):
        memcached = ('get_template_counts',
                     'get_template_for_cluster',
                     'similarity',
                     'get_best_channel',
                     'get_best_channels',
                     'get_probe_depth',
                     )
        cached = ('_get_waveforms',
                  '_get_template_waveforms',
                  '_get_features',
                  '_get_template_features',
                  '_get_amplitudes',
                  '_get_correlograms',
                  )
        _cache_methods(self, memcached, cached)

    def _set_supervisor(self):
        # Load the new cluster id.
        new_cluster_id = self.context.load('new_cluster_id'). \
            get('new_cluster_id', None)
        cluster_groups = self.model.get_metadata('group')
        supervisor = Supervisor(self.model.spike_clusters,
                                similarity=self.similarity,
                                cluster_groups=cluster_groups,
                                new_cluster_id=new_cluster_id,
                                context=self.context,
                                )

        # Load the non-group metadata from the model to the cluster_meta.
        for name in self.model.metadata_fields:
            if name == 'group':
                continue
            values = self.model.get_metadata(name)
            for cluster_id, value in values.items():
                supervisor.cluster_meta.set(name, [cluster_id], value,
                                            add_to_stack=False)

        @supervisor.connect
        def on_create_cluster_views():
            supervisor.add_column(self.get_best_channel, name='channel')
            supervisor.add_column(self.get_probe_depth, name='depth')

            @supervisor.actions.add(shortcut='shift+ctrl+k')
            def split_init(cluster_ids=None):
                """Split a cluster according to the original templates."""
                if cluster_ids is None:
                    cluster_ids = supervisor.selected
                s = supervisor.clustering.spikes_in_clusters(cluster_ids)
                supervisor.split(s, self.model.spike_templates[s])

        # Save.
        @supervisor.connect
        def on_request_save(spike_clusters, groups, *labels):
            """Save the modified data."""
            # Save the clusters.
            self.model.save_spike_clusters(spike_clusters)
            # Save cluster metadata.
            for name, values in labels:
                self.model.save_metadata(name, values)

        return supervisor

    def _set_selector(self):
        def spikes_per_cluster(cluster_id):
            return self.supervisor.clustering.spikes_per_cluster[cluster_id]
        return Selector(spikes_per_cluster)

    def _add_view(self, gui, view, name=None):
        if 'name' in inspect.getargspec(view.attach).args:
            view.attach(gui, name=name)
        else:
            view.attach(gui)
        self.emit('add_view', gui, view)
        return view

    # Model methods
    # -------------------------------------------------------------------------

    def get_template_counts(self, cluster_id):
        """Return a histogram of the number of spikes in each template for
        a given cluster."""
        spike_ids = self.supervisor.clustering.spikes_per_cluster[cluster_id]
        st = self.model.spike_templates[spike_ids]
        return np.bincount(st, minlength=self.model.n_templates)

    def get_template_for_cluster(self, cluster_id):
        """Return the largest template associated to a cluster."""
        spike_ids = self.supervisor.clustering.spikes_per_cluster[cluster_id]
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
                return float(sims[cj])
            temp_j = np.nonzero(self.get_template_counts(cj))[0]
            return float(np.max(sims[temp_j]))

        out = [(cj, _sim_ij(cj))
               for cj in self.supervisor.clustering.cluster_ids]
        # NOTE: hard-limit to 100 for performance reasons.
        return sorted(out, key=itemgetter(1), reverse=True)[:100]

    def get_best_channel(self, cluster_id):
        """Return the best channel of a given cluster."""
        template_id = self.get_template_for_cluster(cluster_id)
        return self.model.get_template(template_id).best_channel

    def get_best_channels(self, cluster_id):
        """Return the best channels of a given cluster."""
        template_id = self.get_template_for_cluster(cluster_id)
        return self.model.get_template(template_id).channel_ids

    def get_probe_depth(self, cluster_id):
        """Return the depth of a cluster."""
        channel_id = self.get_best_channel(cluster_id)
        return self.model.channel_positions[channel_id][1]

    # Waveforms
    # -------------------------------------------------------------------------

    def _get_waveforms(self, cluster_id):
        """Return a selection of waveforms for a cluster."""
        pos = self.model.channel_positions
        spike_ids = self.selector.select_spikes([cluster_id],
                                                self.n_spikes_waveforms,
                                                self.batch_size_waveforms,
                                                )
        channel_ids = self.get_best_channels(cluster_id)
        data = self.model.get_waveforms(spike_ids, channel_ids)
        return Bunch(data=data,
                     channel_ids=channel_ids,
                     channel_positions=pos[channel_ids],
                     )

    def _get_mean_waveforms(self, cluster_id):
        b = self._get_waveforms(cluster_id)
        b.data = b.data.mean(axis=0)[np.newaxis, ...]
        b['alpha'] = 1.
        return b

    def _get_template_waveforms(self, cluster_id):
        """Return the waveforms of the templates corresponding to a cluster."""
        pos = self.model.channel_positions
        count = self.get_template_counts(cluster_id)
        template_ids = np.nonzero(count)[0]
        count = count[template_ids]
        # Get local channels.
        channel_ids = self.get_best_channels(cluster_id)
        # Get masks.
        masks = count / float(count.max())
        masks = np.tile(masks.reshape((-1, 1)), (1, len(channel_ids)))
        # Get the mean amplitude for the cluster.
        mean_amp = self._get_amplitudes(cluster_id).y.mean()
        # Get all templates from which this cluster stems from.
        templates = [self.model.get_template(template_id)
                     for template_id in template_ids]
        data = np.stack([b.template * mean_amp for b in templates], axis=0)
        cols = np.stack([b.channel_ids for b in templates], axis=0)
        assert data.shape[1] == cols.shape[1]
        waveforms = from_sparse(data, cols, channel_ids)
        return Bunch(data=waveforms,
                     channel_ids=channel_ids,
                     channel_positions=pos[channel_ids],
                     masks=masks,
                     alpha=1.,
                     )

    def add_waveform_view(self, gui):
        v = WaveformView(waveforms=self._get_waveforms,
                         )
        v = self._add_view(gui, v)

        v.actions.separator()

        @v.actions.add(shortcut='w')
        def toggle_templates():
            f, g = self._get_waveforms, self._get_template_waveforms
            v.waveforms = f if v.waveforms == g else g
            v.on_select()

        @v.actions.add(shortcut='m')
        def toggle_mean_waveforms():
            f, g = self._get_waveforms, self._get_mean_waveforms
            v.waveforms = f if v.waveforms == g else g
            v.on_select()

        return v

    # Features
    # -------------------------------------------------------------------------

    def _get_spike_ids(self, cluster_id=None, load_all=None):
        nsf = self.n_spikes_features
        if cluster_id is None:
            # Background points.
            ns = self.model.n_spikes
            return np.arange(0, ns, max(1, ns // nsf))
        else:
            # Load all spikes from the cluster if load_all is True.
            n = nsf if not load_all else None
            return self.selector.select_spikes([cluster_id], n)

    def _get_spike_times(self, cluster_id=None):
        spike_ids = self._get_spike_ids(cluster_id)
        return Bunch(data=self.model.spike_times[spike_ids],
                     lim=(0., self.model.duration))

    def _get_features(self, cluster_id=None, channel_ids=None, load_all=None):
        spike_ids = self._get_spike_ids(cluster_id, load_all=load_all)
        # Use the best channels only if a cluster is specified and
        # channels are not specified.
        if cluster_id is not None and channel_ids is None:
            channel_ids = self.get_best_channels(cluster_id)
        data = self.model.get_features(spike_ids, channel_ids)
        return Bunch(data=data,
                     spike_ids=spike_ids,
                     channel_ids=channel_ids,
                     )

    def add_feature_view(self, gui):
        v = FeatureView(features=self._get_features,
                        attributes={'time': self._get_spike_times}
                        )
        return self._add_view(gui, v)

    # Template features
    # -------------------------------------------------------------------------

    def _get_template_features(self, cluster_ids):
        assert len(cluster_ids) == 2
        clu0, clu1 = cluster_ids

        s0 = self._get_spike_ids(clu0)
        s1 = self._get_spike_ids(clu1)

        n0 = self.get_template_counts(clu0)
        n1 = self.get_template_counts(clu1)

        t0 = self.model.get_template_features(s0)
        t1 = self.model.get_template_features(s1)

        x0 = np.average(t0, weights=n0, axis=1)
        y0 = np.average(t0, weights=n1, axis=1)

        x1 = np.average(t1, weights=n0, axis=1)
        y1 = np.average(t1, weights=n1, axis=1)

        return Bunch(x0=x0, y0=y0, x1=x1, y1=y1,
                     data_bounds=(min(x0.min(), x1.min()),
                                  min(y0.min(), y1.min()),
                                  max(y0.max(), y1.max()),
                                  max(y0.max(), y1.max()),
                                  ),
                     )

    def add_template_feature_view(self, gui):
        v = TemplateFeatureView(coords=self._get_template_features,
                                )
        return self._add_view(gui, v, name='TemplateFeatureView')

    # Traces
    # -------------------------------------------------------------------------

    def _get_traces(self, interval):
        """Get traces and spike waveforms."""
        k = self.model.n_samples_templates
        m = self.model
        c = m.channel_vertical_order

        traces_interval = select_traces(m.traces, interval,
                                        sample_rate=m.sample_rate)
        # Reorder vertically.
        traces_interval = traces_interval[:, c]
        out = Bunch(data=traces_interval)
        out.waveforms = []

        def gbc(cluster_id):
            return c[self.get_best_channels(cluster_id)]

        for b in _iter_spike_waveforms(interval=interval,
                                       traces_interval=traces_interval,
                                       model=self.model,
                                       supervisor=self.supervisor,
                                       color_selector=self.color_selector,
                                       n_samples_waveforms=k,
                                       get_best_channels=gbc,
                                       show_all_spikes=self._show_all_spikes,
                                       ):
            i = b.spike_id
            # Compute the residual: waveform - amplitude * template.
            residual = b.copy()
            template_id = m.spike_templates[i]
            template = m.get_template(template_id).template[:, b.channel_ids]
            amplitude = m.amplitudes[i]
            residual.data = residual.data - amplitude * template
            out.waveforms.extend([b, residual])
        return out

    def _jump_to_spike(self, view, delta=+1):
        """Jump to next or previous spike from the selected clusters."""
        m = self.model
        cluster_ids = self.supervisor.selected
        if len(cluster_ids) == 0:
            return
        spc = self.supervisor.clustering.spikes_per_cluster
        spike_ids = spc[cluster_ids[0]]
        spike_times = m.spike_times[spike_ids]
        ind = np.searchsorted(spike_times, view.time)
        n = len(spike_times)
        view.go_to(spike_times[(ind + delta) % n])

    def add_trace_view(self, gui):
        m = self.model
        v = TraceView(traces=self._get_traces,
                      n_channels=m.n_channels,
                      sample_rate=m.sample_rate,
                      duration=m.duration,
                      channel_labels=self.model.channel_vertical_order,
                      )
        self._add_view(gui, v)

        v.actions.separator()

        @v.actions.add(shortcut='alt+pgdown')
        def go_to_next_spike():
            """Jump to the next spike from the first selected cluster."""
            self._jump_to_spike(v, +1)

        @v.actions.add(shortcut='alt+pgup')
        def go_to_previous_spike():
            """Jump to the previous spike from the first selected cluster."""
            self._jump_to_spike(v, -1)

        v.actions.separator()

        @v.actions.add(shortcut='alt+s')
        def toggle_highlighted_spikes():
            """Toggle between showing all spikes or selected spikes."""
            self._show_all_spikes = not self._show_all_spikes
            v.set_interval(force_update=True)

        return v

    # Correlograms
    # -------------------------------------------------------------------------

    def _get_correlograms(self, cluster_ids, bin_size, window_size):
        spike_ids = self.selector.select_spikes(cluster_ids,
                                                self.n_spikes_correlograms,
                                                subset='random',
                                                )
        st = self.model.spike_times[spike_ids]
        sc = self.supervisor.clustering.spike_clusters[spike_ids]
        return correlograms(st,
                            sc,
                            sample_rate=self.model.sample_rate,
                            cluster_ids=cluster_ids,
                            bin_size=bin_size,
                            window_size=window_size,
                            )

    def add_correlogram_view(self, gui):
        m = self.model
        v = CorrelogramView(correlograms=self._get_correlograms,
                            sample_rate=m.sample_rate,
                            )
        return self._add_view(gui, v)

    # Amplitudes
    # -------------------------------------------------------------------------

    def _get_amplitudes(self, cluster_id):
        n = self.n_spikes_amplitudes
        m = self.model
        spike_ids = self.selector.select_spikes([cluster_id], n)
        x = m.spike_times[spike_ids]
        y = m.amplitudes[spike_ids]
        return Bunch(x=x, y=y, data_bounds=(0., 0., m.duration, y.max()))

    def add_amplitude_view(self, gui):
        v = ScatterView(coords=self._get_amplitudes,
                        )
        return self._add_view(gui, v, name='AmplitudeView')

    # GUI
    # -------------------------------------------------------------------------

    def create_gui(self, **kwargs):
        gui = GUI(name=self.gui_name,
                  subtitle=self.model.dat_path,
                  config_dir=self.config_dir,
                  **kwargs)

        self.supervisor.attach(gui)

        self.add_waveform_view(gui)
        self.add_trace_view(gui)
        if self.model.features is not None:
            self.add_feature_view(gui)
        if self.model.template_features is not None:
            self.add_template_feature_view(gui)
        self.add_correlogram_view(gui)
        if self.model.amplitudes is not None:
            self.add_amplitude_view(gui)

        # Save the memcache when closing the GUI.
        @gui.connect_
        def on_close():
            self.context.save_memcache()

        self.emit('gui_ready', gui)

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
        def gui(ctx, params_path):
            """Launch the Template GUI on a params.py file."""

            # Create a `phy.log` log file with DEBUG level.
            _add_log_file(op.join(op.dirname(params_path), 'phy.log'))

            create_app()

            params = _read_python(params_path)
            params['dtype'] = np.dtype(params['dtype'])

            _run_cmd('_run(params)', ctx, globals(), locals())

        @cli.command('template-describe')
        @click.argument('params-path', type=click.Path(exists=True))
        def describe(params_path):
            """Describe a template dataset."""
            params = _read_python(params_path)
            TemplateModel(**params).describe()
