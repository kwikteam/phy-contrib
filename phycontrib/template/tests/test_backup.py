# -*- coding: utf-8 -*-

"""Test the backup plugin."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
import os
import os.path as op
import shutil

import numpy as np
from numpy.testing import assert_equal as ae

from phy.utils._misc import _read_text

from ..backup import _load_backup

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_backup_1(tempdir, template_controller_clean):
    controller = template_controller_clean
    s = controller.supervisor
    sc_path = op.join(tempdir, 'spike_clusters.npy')
    cg_path = op.join(tempdir, 'cluster_group.tsv')

    s.merge([1, 2])
    s.move('good', [64])
    s.split([10, 20, 30, 40])
    s.undo()
    s.split([10, 20, 30, 40, 50])
    s.move('noise', [64])
    s.move('mua', [65])
    s.undo()
    s.redo()
    s.merge([3, 4])

    sc_mem = s.clustering.spike_clusters
    s.save()

    # Save the files.
    sc = np.load(sc_path).squeeze()
    ae(sc, sc_mem)
    cg = _read_text(cg_path)

    # Reset the directory.
    shutil.rmtree(op.join(tempdir, '.phy'))
    os.remove(sc_path)
    os.remove(cg_path)

    # Load the backup.
    assert not op.exists(sc_path)
    _load_backup(op.join(tempdir, 'params.py'))
    assert op.exists(sc_path)

    sc_bak = np.load(sc_path).squeeze()
    ae(sc_bak, sc)

    cg_bak = _read_text(cg_path)
    assert cg_bak == cg
