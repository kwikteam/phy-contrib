# -*- coding: utf-8 -*-

"""Testing the Template model."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
import os.path as op

from pytest import fixture

from phy.utils._misc import _read_python
from ..gui import TemplateController
from phycontrib import _copy_gui_state

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

@fixture
def controller(tempdir, template_model):
    _copy_gui_state('TemplateGUI', 'template', config_dir=tempdir)
    c = TemplateController(model=template_model, config_dir=tempdir)
    return c


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

    s.actions.split_init()
    s.actions.next()
    clu = s.selected[0]
    s.actions.label('some_field', 3)

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

    # Check that the data has been updated.
    assert s.get_labels('some_field')[clu - 1] is None
    assert s.get_labels('some_field')[clu] == '3'

    assert s.cluster_meta.get('group', clu_moved) == 'good'
    for clu in clu_to_merge:
        assert clu not in s.clustering.cluster_ids
    assert clu_merged in s.clustering.cluster_ids
    gui.close()
