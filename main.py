from src.funcs import *
from src.utils import load_data
from src.simulator import Simulator
from src.simulator import World
from collections import namedtuple
import os
import sys

file = "./datasets/a.txt"
config = load_data(file)

world = World(config=config)
world.build()

simulator = Simulator(config=config,
                      world=world)

print('done')
# simulator.start_loop()





