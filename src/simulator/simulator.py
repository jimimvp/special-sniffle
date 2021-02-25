import tqdm
from .world import World

class Simulator:

    def __init__(self, *, config, world):
        self.config = config
        self.world = world

    def start_loop(self):
    
        for t in tqdm.tqdm(range(self.config.simulation_duration)):
            self.world.step()
    
