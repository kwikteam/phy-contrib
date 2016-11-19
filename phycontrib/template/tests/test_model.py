# -*- coding: utf-8 -*-

"""Testing the Template model."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
import os.path as op
import shutil

from pytest import fixture

from phy.utils._misc import _read_python

from ..model import TemplateModel
from phycontrib.utils.testing import download_test_file

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

_FILES = ['template/params.py',
          'template/amplitudes.npy',
          'template/pc_feature_ind.npy',
          'template/spike_clusters.npy',
          'template/template_features.npy',
          'template/whitening_mat_inv.npy',
          'template/channel_map.npy',
          'template/pc_features.npy',
          'template/spike_templates.npy',
          'template/templates_ind.npy',
          'template/whitening_mat.npy',
          'template/channel_positions.npy',
          'template/sim_binary.dat',
          'template/spike_times.npy',
          'template/templates.npy',
          'template/similar_templates.npy',
          'template/template_feature_ind.npy',
          'template/templates_unw.npy',
          ]


@fixture
def template_path(tempdir):
    # Download the dataset.
    paths = list(map(download_test_file, _FILES))
    # Copy the dataset to a temporary directory.
    for path in paths:
        shutil.copy(path,
                    op.join(tempdir, op.basename(path)))
    template_path = op.join(tempdir, op.basename(paths[0]))
    return template_path


def test_model_1(tempdir, template_path):
    params = _read_python(template_path)
    params['dat_path'] = template_path
    model = TemplateModel(**params)
    print(model)
