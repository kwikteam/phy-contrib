# -*- coding: utf-8 -*-

"""Testing the Kwik GUI."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging

from click.testing import CliRunner
from pytest import fixture

from phy.gui.qt import Qt
from phy.utils.cli import phy

from ..gui import KwikController, KwikGUIPlugin
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


@fixture
def runner():
    runner = CliRunner()
    KwikGUIPlugin().attach_to_cli(phy)
    return runner


def test_kwik_describe(runner, kwik_path):
    res = runner.invoke(phy, ['kwik-describe', kwik_path])
    res.exit_code == 0
    assert 'main*' in res.output


def test_kwik_gui(tempdir, qtbot, kwik_path):
    controller = KwikController(kwik_path,
                                config_dir=tempdir,
                                cache_dir=tempdir,
                                )
    gui = controller.create_gui()
    qtbot.addWidget(gui)
    gui.show()
    qtbot.waitForWindowShown(gui)

    qtbot.keyPress(gui, Qt.Key_Down)
    qtbot.keyPress(gui, Qt.Key_Down)

    gui.close()
