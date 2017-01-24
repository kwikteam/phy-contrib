# -*- coding: utf-8 -*-

"""Backup plugin."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import csv
from datetime import datetime
import logging
import os
import os.path as op
import shutil

import click

from phy import IPlugin
from phy.gui.qt import QTimer
from phy.utils._misc import _read_python


logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Utility functions
#------------------------------------------------------------------------------

def _now():
    return datetime.now().strftime("%Y%m%d%H%M%S")


def _backup(dir_path):
    backup_dir = op.join(dir_path, '.phy-backup')
    time = _now()

    filenames = ['spike_clusters.npy', 'cluster_group.tsv']
    for filename in filenames:
        path = op.join(dir_path, filename)
        path_copy = op.join(backup_dir, '%s.bak-%s' % (filename, time))

        if op.exists(path):
            logger.log(5, "Backup %s.", op.basename(path_copy))
            shutil.copy(path, path_copy)


def _delete_old_backup(backup_dir, max_n_files):
    filenames = sorted(os.listdir(backup_dir))
    if len(filenames) < max_n_files:
        return
    for filename in filenames[:-max_n_files]:
        logger.log(5, "Delete %s.", filename)
        os.remove(op.join(backup_dir, filename))


def _write_row(log_path, row):
    row = (_now(),) + row
    with open(log_path, 'a') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(row)


def _load_rows(log_path):
    with open(log_path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            yield row[1:]


def _parse_arg(s):
    if ',' in s:
        return list(map(int, s.split(',')))
    elif s.isdigit():
        return [int(s)]
    else:
        return s


def _replay_actions(log_path, controller):
    c = controller.supervisor
    for row in _load_rows(log_path):
        a = row[0]
        ids = _parse_arg(row[1])
        to = _parse_arg(row[2])

        if a == 'merge':
            c.merge(ids, to[0])
        elif a == 'assign':
            c.split(ids, to)
        elif a == 'metadata_group':
            c.move(to, ids)
        elif a == 'undo':
            c.undo()
        elif a == 'redo':
            c.redo()
    c.save()


def _load_backup(params_path):
    params = _read_python(params_path)
    params['dat_path'] = op.join(op.dirname(params_path), params['dat_path'])
    dat_path = params['dat_path']
    assert dat_path
    dir_path = op.dirname(op.realpath(op.expanduser(dat_path)))
    dir_path = params.get('dir_path', dir_path)

    # Make sure we start from a clean state.
    if op.exists(op.join(dir_path, '.phy')):
        logger.warn("Please delete the .phy subfolder before loading "
                    "the backup.")
        return
    for filename in ('spike_clusters.npy', 'cluster_group.tsv'):
        if op.exists(op.join(dir_path, filename)):
            logger.warn("Please delete the %s file "
                        "before loading the backup.", filename)
            return

    # Create the controller.
    from phycontrib.template import TemplateController
    controller = TemplateController(config_dir=op.dirname(params_path),
                                    **params)

    backup_dir = op.join(dir_path, '.phy-backup')

    # Get the log path.
    log_path = op.join(backup_dir, 'history.tsv')
    if not op.exists(log_path):
        logger.warn("The file `%s` doesn't exist.", log_path)
        return
    _replay_actions(log_path, controller)


#------------------------------------------------------------------------------
# Plugin
#------------------------------------------------------------------------------

class BackupPlugin(IPlugin):
    max_n_files = 10  # max number of backup files to keep
    delay = 120  # in seconds

    def _tick(self):
        """Backup and delete old backup. Called every `delay` seconds."""
        _backup(self.dir_path)
        _delete_old_backup(self.backup_path, self.max_n_files)

    def _set_timer(self, controller):
        @controller.connect
        def on_gui_ready(gui):
            if controller.gui_name != 'TemplateGUI':
                return

            # Backup every `delay` seconds.
            timer = QTimer(gui)
            timer.timeout.connect(self._tick)
            timer.start(self.delay * 1000)  # in milliseconds.

    def _set_history_logger(self, controller):
        # Log the actions in a machine-readable way.
        log_path = op.join(self.backup_dir, 'history.tsv')

        # Log clustering actions.
        @controller.supervisor.clustering.connect
        def on_cluster(up):
            if up.history:
                row = (up.history.lower(), None, None)
            elif up.description == 'merge':
                row = ('merge',
                       ','.join(map(str, up.deleted)),
                       up.added[0])
            else:
                row = ('assign',
                       ','.join(map(str, up.spike_ids)),
                       ','.join(map(str, up.spike_clusters)))

            _write_row(log_path, row)

        # Log cluster_meta actions.
        @controller.supervisor.cluster_meta.connect  # noqa
        def on_cluster(up):
            if up.history:
                row = (up.history.lower(), None, None)
            else:
                row = (up.description,
                       ','.join(map(str, up.metadata_changed)),
                       up.metadata_value,
                       )

            _write_row(log_path, row)

    def attach_to_controller(self, controller):
        self.dir_path = controller.model.dir_path
        self.backup_dir = op.join(self.dir_path, '.phy-backup')
        if not op.exists(self.backup_dir):
            os.mkdir(self.backup_dir)
        self._set_timer(controller)
        self._set_history_logger(controller)

    def attach_to_cli(self, cli):
        @cli.command('template-load-backup')
        @click.argument('params-path', type=click.Path(exists=True))
        def load_backup(params_path):
            """Load the backup."""
            _load_backup(params_path)
