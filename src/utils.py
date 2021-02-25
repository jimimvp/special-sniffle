

from collections import namedtuple
from .simulator.world import Street, Car


class Config(object):
    pass


    def __str__(self) -> str:
        for k,v in self.__dict__.items():
            print(f"{k}:{v}")



Street = namedtuple("Street", field_names=["name", "intersection_start", "intersection_end", "T"])
Car = namedtuple("Car", field_names=["total_route"])




def submission_string(sol, world):
    
    solstr = []
    solstr.append(f"{len(world.intersections)}")
    i = 0
    for k, intersection in world.intersections.items():
        traffic_lights = list(map(lambda x: int(x), sol[i:i+len(intersection.in_streets)])) 
        solstr.append(f"{k}\n{len(intersection.in_streets)}")
        for s, t in zip(intersection.in_streets, traffic_lights):
            solstr.append(f"{s.name} {max(int(t), 0)}")

    return "\n".join(solstr)




def load_data(file):

    config = Config()

    with open(file, "r") as f:
        lines = f.readlines()
        
        config.simulation_duration, config.num_intersections, config.num_streets, config.num_cars, config.car_score = list(map(lambda x: int(x), lines[0].strip().split(" ")))
        config.streets = []

        lines = lines[1:]
        for i in range(config.num_streets):

            l = lines[i].strip()
            start, end, sid,  L = l.split(" ") 
            config.streets.append(Street(sid, int(start), int(end), int(L)))

        lines = lines[config.num_streets:]
        config.cars = []
        for i in range(config.num_cars):
            config.cars.append(Car(lines[i].strip().split(" ")[1:]))

    return config
