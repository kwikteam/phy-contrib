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
from phycontrib import _copy_gui_state

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

@fixture
def controller(tempdir):
    _copy_gui_state('KwikGUI', 'kwik', config_dir=tempdir)
    # Download the dataset.
    paths = list(map(download_test_file, ('kwik/hybrid_10sec.kwik',
                                          'kwik/hybrid_10sec.kwx',
                                          'kwik/hybrid_10sec.dat')))
    # Copy the dataset to a temporary directory.
    for path in paths:
        shutil.copy(path, op.join(tempdir, op.basename(path)))
    kwik_path = op.join(tempdir, op.basename(paths[0]))
    c = KwikController(kwik_path)
    return c


@fixture
def runner():
    runner = CliRunner()
    KwikGUIPlugin().attach_to_cli(phy)
    return runner


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_kwik_describe(runner, controller):
    res = runner.invoke(phy, ['kwik-describe', controller.model.kwik_path])
    res.exit_code == 0
    assert 'main*' in res.output


def test_gui_1(qtbot, tempdir, controller):
    gui = controller.create_gui()
    s = controller.supervisor
    gui.show()
    qtbot.waitForWindowShown(gui)

    wv = gui.list_views('WaveformView')[0]
    tv = gui.list_views('TraceView')[0]

    tv.actions.go_to_next_spike()

    s.actions.next()
    qtbot.wait(100)
    clu_moved = s.selected[0]
    s.actions.move_best_to_good()
    assert len(s.selected) == 1
    s.actions.next()
    clu_to_merge = s.selected
    assert len(clu_to_merge) == 2
    # Ensure the template feature view is updated.
    qtbot.wait(100)
    s.actions.merge()
    clu_merged = s.selected[0]
    s.actions.move_all_to_mua()

    s.actions.next()
    clu = s.selected[0]

    wv.actions.toggle_mean_waveforms()

    tv.actions.toggle_highlighted_spikes()
    tv.actions.go_to_next_spike()
    tv.actions.go_to_previous_spike()

    s.save()
    gui.close()

    # Create a new controller and a new GUI with the same data.
    controller = KwikController(controller.model.kwik_path, config_dir=tempdir)

    gui = controller.create_gui()
    s = controller.supervisor
    gui.show()
    qtbot.waitForWindowShown(gui)

    assert s.cluster_meta.get('group', clu_moved) == 'good'
    for clu in clu_to_merge:
        assert clu not in s.clustering.cluster_ids
    assert clu_merged in s.clustering.cluster_ids
    gui.close()


def test_kwik_gui_2(qtbot, controller):
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
    qtbot.keyPress(gui, Qt.Key_Space)
    # Recluster.
    qtbot.keyPress(gui, Qt.Key_Colon)
    for char in 'RECLUSTER':
        qtbot.keyPress(gui, getattr(Qt, 'Key_' + char))
    qtbot.keyPress(gui, Qt.Key_Enter)
    qtbot.keyPress(gui, Qt.Key_S, modifier=Qt.ControlModifier)

    gui.close()
