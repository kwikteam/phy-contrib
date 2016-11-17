# -*- coding: utf-8 -*-

"""Testing the Kwik GUI."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
import os.path as op
import shutil

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
def kwik_path(tempdir):
    # Download the dataset.
    paths = list(map(download_test_file, ('kwik/hybrid_10sec.kwik',
                                          'kwik/hybrid_10sec.kwx',
                                          'kwik/hybrid_10sec.dat')))
    # Copy the dataset to a temporary directory.
    for path in paths:
        shutil.copy(path, op.join(tempdir, op.basename(path)))
    kwik_path = op.join(tempdir, op.basename(paths[0]))
    return kwik_path


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
    qtbot.keyPress(gui, Qt.Key_Space)
    qtbot.keyPress(gui, Qt.Key_G)
    qtbot.keyPress(gui, Qt.Key_Space)
    qtbot.keyPress(gui, Qt.Key_G, modifier=Qt.AltModifier)
    qtbot.keyPress(gui, Qt.Key_Z)
    qtbot.keyPress(gui, Qt.Key_N, modifier=Qt.AltModifier)
    qtbot.keyPress(gui, Qt.Key_S, modifier=Qt.ControlModifier)

    gui.close()
