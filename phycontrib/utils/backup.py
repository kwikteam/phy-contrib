from datetime import datetime
import logging
import os
import os.path as op
import shutil

from phy import IPlugin
from phy.gui.qt import QTimer


logger = logging.getLogger(__name__)


def _backup(dir_path):
    backup_dir = op.join(dir_path, '.phy-backup')
    if not op.exists(backup_dir):
        os.mkdir(backup_dir)
    time = datetime.now().strftime("%Y%m%d%H%M%S")

    filenames = ['spike_clusters.npy', 'cluster_group.tsv']
    for filename in filenames:
        path = op.join(dir_path, filename)
        path_copy = op.join(backup_dir, '%s.bak-%s' % (filename, time))

        if op.exists(path):
            logger.log(5, "Backup %s.", op.basename(path_copy))
            shutil.copy(path, path_copy)


def _delete_old_backup(dir_path, max_n_files):
    backup_dir = op.join(dir_path, '.phy-backup')
    filenames = sorted(os.listdir(backup_dir))
    if len(filenames) < max_n_files:
        return
    for filename in filenames[:-max_n_files]:
        logger.log(5, "Delete %s.", filename)
        os.remove(op.join(backup_dir, filename))


class BackupPlugin(IPlugin):
    max_n_files = 10
    delay = 120

    def attach_to_cli(self, cli):
        pass

    def _tick(self):
        """Backup and delete old backup. Called every `delay` seconds."""
        _backup(self.dir_path)
        _delete_old_backup(self.dir_path, self.max_n_files)

    def attach_to_controller(self, controller):
        @controller.connect
        def on_gui_ready(gui):
            if controller.gui_name != 'TemplateGUI':
                return
            self.dir_path = controller.model.dir_path
            timer = QTimer(gui)
            timer.timeout.connect(self._tick)
            timer.start(self.delay * 1000)  # in milliseconds.
