from src.funcs import *
from src.utils import load_data
from src.simulator import Simulator
from src.simulator import World
from src.opt.cem import cem_improved
from collections import namedtuple
import numpy as np
import jax


file = "./datasets/a.txt"
config = load_data(file)

world = World(config=config)
world.build()

d = sum([len(intersection.in_streets) for _, intersection in world.intersections.items()])
traffic_lights = np.random.randint(0, 10, d)
i = 0
for _, intersection in world.intersections.items():
    intersection.traffic_lights = traffic_lights[i:i+len(intersection.in_streets)]
    i+=len(intersection.in_streets)

simulator = Simulator(config=config,
                      world=world)
simulator.start_loop(verbose=True)

# run CEM optimization

seed = np.random.randint(0, 1000000)
key = jax.random.PRNGKey(seed)

key, _ = jax.random.split(key)

d = sum([len(intersection.in_streets) for _, intersection in world.intersections.items()])
mu, std, sol = cem_improved(key, config, 100, d, 7, num_elites=10, max_keep_elites=10)