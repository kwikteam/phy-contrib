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
from phy.utils.testing import captured_output

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


def test_model_1(template_model):
    with captured_output() as (stdout, stderr):
        template_model.describe()
    out = stdout.getvalue()
    print(out)
    assert 'sim_binary.dat' in out
    assert '(300000, 32)' in out
    assert '64' in out


def test_model_2(template_model):
    m = template_model
    tmp = m.get_template(3)
    channels = tmp.channels
    spike_ids = m.spikes_in_template(3)

    w = m.get_waveforms(spike_ids, channels)
    assert w.shape == (len(spike_ids), tmp.template.shape[0], len(channels))

    f = m.get_features(spike_ids, channels)
    assert f.shape == (len(spike_ids), len(channels), 3)

    tf = m.get_template_features(spike_ids, channels)
    assert tf.shape == (len(spike_ids), len(channels))
    print(tf)
