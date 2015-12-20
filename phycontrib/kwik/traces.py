# -*- coding: utf-8 -*-

"""Raw data readers."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
import os.path as op

from six import string_types
import numpy as np

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Raw data readers
#------------------------------------------------------------------------------

def _dat_n_samples(filename, dtype=None, n_channels=None):
    assert dtype is not None
    item_size = np.dtype(dtype).itemsize
    n_samples = op.getsize(filename) // (item_size * n_channels)
    assert n_samples >= 0
    return n_samples


def _concat_dask(arrs):
    from dask.array import concatenate, from_array
    return concatenate([from_array(arr, chunks=arr.shape) for arr in arrs],
                       axis=0)


def read_kwd(filename):
    """Read all traces in a `.kwd` file."""
    from h5py import File

    f = File(filename, mode='r')
    recs = sorted([name for name in f['/recordings']])
    arrs = [f['/recordings/%s/data' % rec] for rec in recs]
    return _concat_dask(arrs)


def _read_dat(filename, dtype=None, shape=None, offset=0, n_channels=None):
    """Read traces from a flat binary `.dat` file.

    The output is a memory-mapped file.

    Parameters
    ----------

    filename : str
        The path to the `.dat` file.
    dtype : dtype
        The NumPy dtype.
    offset : 0
        The header size.
    n_channels : int
        The number of channels in the data.
    shape : tuple (optional)
        The array shape. Typically `(n_samples, n_channels)`. The shape is
        automatically computed from the file size if the number of channels
        and dtype are specified.

    """
    if shape is None:
        assert n_channels > 0
        n_samples = _dat_n_samples(filename, dtype=dtype,
                                   n_channels=n_channels)
        shape = (n_samples, n_channels)
    return np.memmap(filename, dtype=dtype, shape=shape,
                     mode='r', offset=offset)


def _read_multiple_dat(filenames, dtype=None, offset=0, n_channels=None):
    return _concat_dask([read_dat(fn, dtype=dtype, offset=offset,
                                  n_channels=n_channels) for fn in filenames])


def read_dat(filename, *args, **kwargs):
    """Read one or multiple dat files."""
    if isinstance(filename, string_types):
        return _read_dat(filename, *args, **kwargs)
    else:
        return _read_multiple_dat(list(filename), *args, **kwargs)
