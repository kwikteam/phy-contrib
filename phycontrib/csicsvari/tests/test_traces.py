# -*- coding: utf-8 -*-

"""Tests of read traces functions."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os.path as op

import numpy as np
from numpy.testing import assert_array_equal as ae

from ..traces import read_dat, _dat_n_samples, read_kwd
from phy.io.mock import artificial_traces


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_read_dat(tempdir):
    n_samples = 100
    n_channels = 10

    arr = artificial_traces(n_samples, n_channels)

    path = op.join(tempdir, 'test')
    arr.tofile(path)
    assert _dat_n_samples(path, dtype=np.float64,
                          n_channels=n_channels) == n_samples
    data = read_dat(path, dtype=arr.dtype, shape=arr.shape)
    ae(arr, data)
    data = read_dat(path, dtype=arr.dtype, n_channels=n_channels)
    ae(arr, data)

    # Test multiple dats.
    arr[:n_samples // 2].tofile(op.join(tempdir, 'test1'))
    arr[n_samples // 2:].tofile(op.join(tempdir, 'test2'))
    ae(read_dat([op.join(tempdir, 'test1'), op.join(tempdir, 'test2')],
       dtype=arr.dtype, n_channels=n_channels), data)


def test_read_kwd(tempdir):
    from h5py import File

    n_samples = 100
    n_channels = 10
    arr = artificial_traces(n_samples, n_channels)
    path = op.join(tempdir, 'test.kwd')

    with File(path, 'w') as f:
        g0 = f.create_group('/recordings/0')
        g1 = f.create_group('/recordings/1')

        arr0 = arr[:n_samples // 2, ...]
        arr1 = arr[n_samples // 2:, ...]

        g0.create_dataset('data', data=arr0)
        g1.create_dataset('data', data=arr1)

    arr1 = read_kwd(path)[...]
    ae(arr1, arr)
