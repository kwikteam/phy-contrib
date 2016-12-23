# -*- coding: utf-8 -*-

"""Testing the Template model."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging

from pytest import fixture

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


def test_gui_1(qtbot, controller):
    gui = controller.create_gui()
    s = controller.supervisor
    gui.show()
    qtbot.waitForWindowShown(gui)
    s.next()
    w = gui.list_views('WaveformView')[0]
    w.actions.toggle_templates()
    w.actions.toggle_mean_waveforms()
    # qtbot.stop()
