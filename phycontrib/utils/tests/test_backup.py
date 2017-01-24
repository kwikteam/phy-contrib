# -*- coding: utf-8 -*-

"""Test the backup plugin."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
import os
import os.path as op
import shutil

from click.testing import CliRunner
import numpy as np
from numpy.testing import assert_equal as ae
from pytest import fixture

from phy.utils.cli import phy
from phy.utils._misc import _read_text

from phycontrib.template.gui import TemplateController

from ..backup import BackupPlugin

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

@fixture
def controller(tempdir, template_model):
    plugins = ['BackupPlugin']
    c = TemplateController(model=template_model, config_dir=tempdir,
                           plugins=plugins)
    return c


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_backup_1(tempdir, controller):
    s = controller.supervisor
    s.merge([1, 2])

    s.save()

    # Save the files.
    sc_path = op.join(tempdir, 'spike_clusters.npy')
    cg_path = op.join(tempdir, 'cluster_group.tsv')
    sc = np.load(sc_path)
    cg = _read_text(cg_path)

    # Reset the directory.
    shutil.rmtree(op.join(tempdir, '.phy'))
    os.remove(sc_path)
    os.remove(cg_path)

    # Load the backup.
    runner = CliRunner()
    BackupPlugin().attach_to_cli(phy)
    path = op.join(tempdir, 'params.py')
    res = runner.invoke(phy, ['template-load-backup', path])
    # assert res.exit_code == 0
    print(res.exc_info)
    print(res.output)
    return

    sc_bak = np.load(op.join(tempdir, 'spike_clusters.npy'))
    ae(sc_bak, sc)

    cg_bak = _read_text(cg_path)
    assert cg_bak == cg
