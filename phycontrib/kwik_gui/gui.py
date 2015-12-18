# -*- coding: utf-8 -*-

"""Kwik GUI."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
import os.path as op
import shutil

import click

from phy import IPlugin
from phy.gui import create_app, create_gui, run_app

from phycontrib.kwik import KwikModel

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Kwik GUI
#------------------------------------------------------------------------------

def _backup(path):
    """Backup a file."""
    path_backup = path + '.bak'
    if not op.exists(path_backup):
        logger.info("Backup `%s`.".format(path_backup))
        shutil.copy(path, path_backup)


#------------------------------------------------------------------------------
# Kwik GUI plugin
#------------------------------------------------------------------------------

class KwikGUIPlugin(IPlugin):
    """Create the `phy cluster-manual` command for Kwik files."""

    def attach_to_cli(self, cli):

        # Create the `phy cluster-manual file.kwik` command.
        @cli.command('cluster-manual')
        @click.argument('path', type=click.Path(exists=True))
        def cluster_manual(path):

            # Open the dataset.
            path = op.realpath(op.expanduser(path))
            _backup(path)
            model = KwikModel(path)

            # Create the Qt application.
            create_app()

            # List of plugins activated by default.
            plugins = ['ContextPlugin',
                       'ManualClusteringPlugin',
                       'WaveformViewPlugin',
                       'FeatureViewPlugin',
                       'CorrelogramViewPlugin',
                       'TraceViewPlugin',
                       'SaveGeometryStatePlugin',
                       ]

            # Create the GUI.
            gui = create_gui(name='KwikGUI',
                             subtitle=model.kwik_path,
                             model=model,
                             plugins=plugins,
                             )

            # Save.
            @gui.connect_
            def on_request_save(spike_clusters, groups):
                groups = {c: g.title() for c, g in groups.items()}
                model.save(spike_clusters, groups)

            # Show the GUI.
            gui.show()

            # Start the Qt event loop.
            run_app()

            # Close the GUI.
            gui.close()
            del gui
