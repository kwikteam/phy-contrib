# -*- coding: utf-8 -*-

"""Save prompt plugin."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from functools import partial

from phy.gui.qt import _prompt, _show_box

from phy import IPlugin
from phy.utils import connect


#------------------------------------------------------------------------------
# Plugin
#------------------------------------------------------------------------------

def prompt_save(supervisor):
    # Show save prompt if an action was done.
    if len(supervisor._global_history) <= 1:
        return
    b = _prompt("Do you want to save your modifications "
                "before quitting?",
                buttons=['save', 'cancel', 'close'],
                title='Save')
    r = _show_box(b)
    if r == 'save':
        supervisor.save()
    elif r == 'cancel':
        return False
    elif r == 'close':
        return


class SavePromptPlugin(IPlugin):
    def attach_to_controller(self, controller):
        connect(partial(prompt_save, controller.supervisor), event='close')
