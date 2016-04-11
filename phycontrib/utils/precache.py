from tqdm import tqdm

from phy import IPlugin


class PrecachePlugin(IPlugin):
    def attach_to_controller(self, controller):
        @controller.connect
        def on_init():
            for clu in tqdm(controller.cluster_ids.tolist(),
                            desc="Precaching data",
                            leave=True,
                            ):
                controller.get_features(clu)
                controller.get_waveforms(clu)
                controller.get_close_clusters(clu)
