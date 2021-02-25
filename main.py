from src.funcs import *
from src.utils import load_data
from src.simulator import Simulator
from src.simulator import World
from collections import namedtuple
import os
import sys
import numpy as np


file = "./datasets/a.txt"
config = load_data(file)

world = World(config=config)
world.build()

solution_dim = sum([len(intersection.in_streets) for _, intersection in world.intersections.items()])
traffic_lights = np.random.randint(0, 10, solution_dim)
i = 0
for _, intersection in world.intersections.items():
    intersection.traffic_lights = traffic_lights[i:i+len(intersection.in_streets)]




simulator = Simulator(config=config,
                      world=world)

print('done')
# simulator.start_loop()





world = World(config=config)
