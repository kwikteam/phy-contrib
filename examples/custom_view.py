"""Custom view plugin.

This plugin adds an interactive matplotlib figure showing the ISI of the
first selected cluster.

To activate the plugin, copy this file to `~/.phy/plugins/` and add this line
to your `~/.phy/phy_config.py`:

```python
c.KwikGUI.plugins = ['CustomView']
```

"""

from phy import IPlugin
import numpy as np
import matplotlib.pyplot as plt


class CustomView(IPlugin):
    def attach_to_controller(self, c):

        # Create the figure when initializing the GUI.
        f, ax = plt.subplots()

        @c.connect
        def on_create_gui(gui):
            # Called when the GUI is created.

            # We add the matplotlib figure to the GUI.
            gui.add_view(f, name='ISI')

            # We connect this function to the "select" event triggered
            # by the GUI at every cluster selection change.
            @gui.connect_
            def on_select(clusters):
                # We clear the figure.
                ax.clear()

                # We compute the ISI.
                spikes = c.spikes_per_cluster(clusters[0])
                ax.hist(np.diff(spikes), bins=50)

                # We update the figure.
                f.canvas.draw()
