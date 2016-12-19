import os.path as op

from tqdm import tqdm

from phy import IPlugin


class PrecachePlugin(IPlugin):
    def attach_to_controller(self, controller):
        # Skip if the cache has already been created.
        if op.exists(op.join(controller.cache_dir, 'done')):
            return

        s = controller.supervisor

        @controller.connect
        def on_gui_ready(gui):
            # Create the cache.
            for clu in tqdm(s.clustering.cluster_ids.tolist(),
                            desc="Precaching data",
                            leave=True,
                            ):
                s.select([clu])
            s.select([])
            # Mark the cache as complete.
            with open(op.join(controller.cache_dir, 'done'), 'w') as f:
                f.write('')
