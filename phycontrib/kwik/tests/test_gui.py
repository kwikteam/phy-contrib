# -*- coding: utf-8 -*-

"""Testing the Kwik GUI."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
import os.path as op

from click.testing import CliRunner
from pytest import fixture

from phy.utils.cli import phy
from ..gui import KwikGUIPlugin
from phycontrib.utils.testing import download_test_file

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

@fixture
def kwik_path():
    path = download_test_file('kwik/hybrid_10sec.kwik')
    download_test_file('kwik/hybrid_10sec.kwx')
    download_test_file('kwik/hybrid_10sec.dat')
    return path


def test_download_data(kwik_path):
    runner = CliRunner()
    KwikGUIPlugin().attach_to_cli(phy)
    res = runner.invoke(phy, ['kwik-describe', kwik_path])
    res.exit_code == 0
    assert 'main*' in res.output
