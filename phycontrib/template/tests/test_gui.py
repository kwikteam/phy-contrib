# -*- coding: utf-8 -*-

"""Testing the Template model."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
import os.path as op

from click.testing import CliRunner
from pytest import fixture

from phy.utils.cli import phy
from phy.gui.qt import Qt
from phy.gui.widgets import Barrier
from phy.utils._misc import _read_python
from phy.utils import connect
from ..gui import TemplateController, TemplateGUIPlugin

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

@fixture
def runner():
    runner = CliRunner()
    TemplateGUIPlugin().attach_to_cli(phy)
    return runner


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_template_controller(template_controller):
    assert template_controller


def test_template_describe(qtbot, runner, tempdir, template_controller):
    path = op.join(tempdir, 'params.py')
    res = runner.invoke(phy, ['template-describe', path])
    res.exit_code == 0
    print(res.output)
    # assert 'main*' in res.output


def _wait_controller(controller):
    mc = controller.supervisor
    b = Barrier()
    connect(b('cluster_view'), event='ready', sender=mc.cluster_view)
    connect(b('similarity_view'), event='ready', sender=mc.similarity_view)
    b.wait()


def test_template_gui_0(qtbot, tempdir, template_controller):
    controller = template_controller
    gui = controller.create_gui()
    gui.show()
    qtbot.waitForWindowShown(gui)
    _wait_controller(controller)
    # qtbot.stop()


def test_template_gui_1(qtbot, tempdir, template_controller):
    controller = template_controller
    gui = controller.create_gui()
    s = controller.supervisor
    gui.show()
    qtbot.waitForWindowShown(gui)
    _wait_controller(controller)

    wv = gui.list_views('WaveformView')[0]
    tv = gui.list_views('TraceView')[0]

    tv.actions.go_to_next_spike()
    s.actions.next()
    s.block()

    s.actions.move_best_to_good()
    s.block()

    assert len(s.selected) == 1
    s.actions.next()
    s.block()

    clu_to_merge = s.selected
    assert len(clu_to_merge) == 2

    s.actions.merge()
    s.block()

    clu_merged = s.selected[0]
    s.actions.move_all_to_mua()
    s.block()
    qtbot.wait(100)

    s.actions.split_init()
    s.block()
    qtbot.wait(100)

    s.actions.next()
    s.block()
    qtbot.wait(100)

    clu = s.selected[0]
    s.actions.label('some_field', 3)
    s.block()
    qtbot.wait(100)

    s.actions.move_all_to_good()
    s.block()

    wv.actions.toggle_templates()
    wv.actions.toggle_mean_waveforms()

    tv.actions.toggle_highlighted_spikes()
    tv.actions.go_to_next_spike()
    tv.actions.go_to_previous_spike()

    s.save()
    gui.close()

    # Create a new controller and a new GUI with the same data.
    params = _read_python(op.join(tempdir, 'params.py'))
    params['dat_path'] = controller.model.dat_path
    controller = TemplateController(config_dir=tempdir,
                                    **params)

    gui = controller.create_gui()
    s = controller.supervisor
    gui.show()
    qtbot.waitForWindowShown(gui)
    _wait_controller(controller)

    # Check that the data has been updated.
    assert s.get_labels('some_field')[clu - 1] is None
    assert s.get_labels('some_field')[clu] == '3'

    assert s.cluster_meta.get('group', clu) == 'good'
    for clu in clu_to_merge:
        assert clu not in s.clustering.cluster_ids
    assert clu_merged in s.clustering.cluster_ids
    gui.close()


def test_template_gui_2(qtbot, template_controller):
    gui = template_controller.create_gui()
    qtbot.addWidget(gui)
    gui.show()
    qtbot.waitForWindowShown(gui)
    _wait_controller(template_controller)

    qtbot.keyPress(gui, Qt.Key_Down)
    qtbot.keyPress(gui, Qt.Key_Down)
    qtbot.keyPress(gui, Qt.Key_Space)
    qtbot.keyPress(gui, Qt.Key_G)
    qtbot.keyPress(gui, Qt.Key_Space)
    qtbot.keyPress(gui, Qt.Key_G, modifier=Qt.AltModifier)
    qtbot.keyPress(gui, Qt.Key_Z)
    qtbot.keyPress(gui, Qt.Key_N, modifier=Qt.AltModifier)
    qtbot.keyPress(gui, Qt.Key_Space)
    qtbot.keyPress(gui, Qt.Key_Enter)
    qtbot.keyPress(gui, Qt.Key_S, modifier=Qt.ControlModifier)

    gui.close()


def test_template_gui_sim(qtbot, template_controller):
    """Ensure that the similarity is refreshed when clusters change."""
    gui = template_controller.create_gui()
    s = template_controller.supervisor
    qtbot.addWidget(gui)
    gui.show()
    qtbot.waitForWindowShown(gui)
    _wait_controller(template_controller)

    s.cluster_view.sort_by('id', 'desc')
    s.actions.next()
    s.block()

    s.similarity_view.sort_by('id', 'desc')
    cl = 63
    assert s.selected == [cl]
    s.actions.next()
    s.block()

    assert s.selected == [cl, cl - 1]
    s.actions.next()
    s.block()

    assert s.selected == [cl, cl - 2]
    s.actions.merge()
    s.block()

    s.actions.next_best()
    s.block()

    s.actions.next()
    s.block()
    assert s.selected == [cl - 1, cl + 1]

    gui.close()
