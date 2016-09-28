import os.path as op

from tqdm import tqdm

from phy import IPlugin


class PrecachePlugin(IPlugin):
    def attach_to_controller(self, controller):
        @controller.connect
        def on_init():
            # Skip if the cache has already been created.
            if op.exists(op.join(controller.cache_dir, 'done')):
                return
            # Create the cache.
            for clu in tqdm(controller.cluster_ids.tolist(),
                            desc="Precaching data",
                            leave=True,
                            ):
                controller.get_features(clu)
                controller.get_waveforms(clu)
                controller.get_close_clusters(clu)
            # Mark the cache as complete.
            with open(op.join(controller.cache_dir, 'done'), 'w') as f:
                f.write('')
