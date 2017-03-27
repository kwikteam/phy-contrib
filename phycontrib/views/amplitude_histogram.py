"""AmplitudeHistogram view plugin.

This plugin adds a matplotlib view showing amplitude histograms for
the selected clusters. The percentage of undetected spikes for selected
clusters are estimated by fitting a gaussian function with a cutoff,
and displayed on the command line.

To activate the plugin, copy this file to `~/.phy/plugins/` and add this line
to your `~/.phy/phy_config.py`:

```python
c.TemplateGUI.plugins = ['AmplitudeHistogram']
```
Luke Shaheen - Laboratory of Brain, Hearing and Behavior Nov 2016
"""

import logging

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import ndtr

from phy import IPlugin
from phy.gui.qt import AsyncCaller, busy_cursor
from phy.utils._color import _spike_colors

logger = logging.getLogger(__name__)


class AmplitudeHistogram(IPlugin):
    def attach_to_controller(self, controller):

        # Create the figure when initializing the GUI.
        plt.rc('xtick', color='w')
        plt.rc('ytick', color='w')
        plt.rc('axes', edgecolor='w')
        f = plt.figure()
        ax = f.add_axes([0.15, 0.1, 0.78, 0.87])
        rect = f.patch
        rect.set_facecolor('k')
        ax.set_axis_bgcolor('k')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.get_yaxis().set_tick_params(direction='out')
        ax.get_xaxis().set_tick_params(direction='out')

        def gaussian(x, a, x0, sigma):
            return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

        def gaussian_cut(x, a, x0, sigma, xcut):
            g = a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
            g[x < xcut] = 0
            return g

        def _update(clusters):
            ax.clear()
            colors = _spike_colors(np.arange(len(clusters)))
            maxs = np.zeros(len(clusters))
            was_fit = np.zeros(len(clusters), dtype=bool)
            for i in range(len(clusters)):
                #plot the amplitude histogram
                coords = controller._get_amplitudes(clusters[i])
                if len(clusters) == 1:
                    colors[i][3] = 1
                num, bins, patches = ax.hist(coords.y, bins=50,
                                             facecolor=colors[i],
                                             edgecolor='none',
                                             orientation='horizontal')

                #fit a gaussian to the histogram
                mean_seed = coords.y.mean()
                mean_seed = bins[np.argmax(num)]  # mode of mean_seed
                bin_steps = np.diff(bins[:2])[0]
                x = bins[:-1] + bin_steps / 2
                next_low_bin = x[0] - bin_steps
                add_points = np.flipud(np.arange(next_low_bin,
                                                 0, -bin_steps))
                x = np.append(add_points, x)
                num = np.append(np.zeros(len(add_points)), num)
                if 0:
                    #old way, gaussian fit
                    popt, pcov = curve_fit(gaussian,
                                           x,
                                           num,
                                           p0=(num.max(),
                                               mean_seed,
                                               2 * coords.y.std()))
                    n_fit = gaussian(x, popt[0], popt[1], popt[2])
                    min_amplitude = coords.y.min()
                    was_fit[i] = True
                else:
                    #new way, cutoff gaussian fit
                    p0 = (num.max(), mean_seed, 2 * coords.y.std(),
                          np.percentile(coords.y, 1))
                    try:
                        popt, pcov = curve_fit(gaussian_cut, x, num, p0=p0,
                                               maxfev=10000)
                        was_fit[i] = True
                    except Exception as e:
                        try:
                            logger.info("Fitting failed with maxfev=10000"
                                        ", trying maxfev=1000000")
                            popt, pcov = curve_fit(gaussian_cut, x, num,
                                                   p0=p0, maxfev=1000000)
                            was_fit[i] = True
                        except Exception as e:
                            logger.info("Fitting error: %s", str(e))
                            was_fit[i] = False
                    if was_fit[i]:
                        n_fit = gaussian_cut(x, popt[0], popt[1],
                                             popt[2], popt[3])
                        min_amplitude = popt[3]
                        n_fit_no_cut = gaussian_cut(x, popt[0], popt[1],
                                                    popt[2], 0)
                        ax.plot(n_fit_no_cut, x, c='w', linestyle='--')
                if was_fit[i]:
                    maxs[i] = n_fit.max()
                    ax.plot(n_fit, x, c='w')
                    # norm area calculated by fit parameters
                    norm_area_ndtr = ndtr((popt[1] - min_amplitude) /
                                          popt[2])
                    percent_missing_ndtr = 100 * (1 - norm_area_ndtr)

                    logger.debug('Cluster %d is missing %.1f of spikes',
                                 clusters[i], percent_missing_ndtr)

            if any(was_fit):
                ax.set_xlim([0, maxs.max() * 1.3])
            xt = ax.get_xticks()
            ax.set_xticks((xt[0], xt[-1]))
            f.canvas.draw()

        @controller.connect
        def on_gui_ready(gui):
            # Called when the GUI is created.
            # We add the matplotlib figure to the GUI.
            gui.add_view(f, name='AmplitudeHistogram')

            # Call on_select() asynchronously after a delay, and set a busy
            # cursor.
            async_caller = AsyncCaller(delay=100)

            @gui.connect_
            def on_select(cluster_ids, **kwargs):
                # Call this function after a delay unless there is another
                # cluster selection in the meantime.
                @async_caller.set
                def update_view():
                    with busy_cursor():
                        _update(cluster_ids)
