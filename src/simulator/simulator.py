import tqdm
from .world import World

class Simulator:

    def __init__(self, config):
        self.config = config
        self.world = World(config)

    def loop(self):
    
        for t in tqdm.tqdm(self.config.simulation_duration):
            pass
    
