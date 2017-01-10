# -*- coding: utf-8 -*-

"""Test fixtures."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os.path as op
import shutil

from pytest import fixture

from phy.utils._misc import _read_python

from ..model import TemplateModel
from phycontrib.utils.testing import download_test_file


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

_FILES = ['template/params.py',
          'template/sim_binary.dat',
          'template/spike_times.npy',
          'template/spike_templates.npy',
          'template/spike_clusters.npy',
          'template/amplitudes.npy',

          'template/cluster_group.tsv',

          'template/channel_map.npy',
          'template/channel_positions.npy',

          'template/templates.npy',
          # 'template/template_ind.npy',
          'template/similar_templates.npy',
          'template/whitening_mat.npy',

          'template/pc_features.npy',
          'template/pc_feature_ind.npy',

          'template/template_features.npy',
          'template/template_feature_ind.npy',

          ]


@fixture
def template_model(tempdir):
    # Download the dataset.
    paths = list(map(download_test_file, _FILES))
    # Copy the dataset to a temporary directory.
    for path in paths:
        shutil.copy(path,
                    op.join(tempdir, op.basename(path)))
    template_path = op.join(tempdir, op.basename(paths[0]))

    params = _read_python(template_path)
    params['dat_path'] = op.join(op.dirname(template_path), params['dat_path'])
    model = TemplateModel(**params)

    return model
