import tqdm
from .world import World

class Simulator:

    def __init__(self, *, config, world):
        self.config = config
        self.world = world

    def start_loop(self, verbose=False):
    
        for t in tqdm.tqdm(range(self.config.simulation_duration)):
            self.world.step()

            if verbose:
                self.world.print_state()
                input()