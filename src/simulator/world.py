import numpy as np

from os import sched_setscheduler


class Street:
    def __init__(self, *, name, intersection_start, intersection_end, T):
        self.name = name
        self.intersection_start = intersection_start
        self.intersection_end = intersection_end
        self.T = T
        self.cars = []
        self.traveling_cars = []
        self.queueing_cars = [] # FIFO
    
    def step(self, t):
        for car in self.traveling_cars:
            car.remaining_traveling_time -= 1
            if car.remaining_traveling_time == 0:
                self.traveling_cars.remove(car)
                self.queueing_cars.append(car)
                car.just_arrived_at_intersection = True

class Intersection:
    def __init__(self, *, id):
        self.identifier = id
        self.in_streets = []
        self.out_streets = []
        
        self.traffic_lights = []
        self.curr_green = 0
        self.inner_counter = 0


    def step(self, t):

        assert len(self.traffic_lights) == len(self.in_streets)        
        if np.all(self.traffic_lights == 0):
            self.curr_green = -1

            return

        while self.traffic_lights[self.curr_green] == 0:
            self.curr_green = (self.curr_green + 1) % len(self.traffic_lights)
        if self.traffic_lights[self.curr_green] == self.inner_counter:
            self.curr_green = (self.curr_green + 1) % len(self.traffic_lights)
            self.inner_counter = 0

        # let car out
        street = self.in_streets[self.curr_green]
        if street.queueing_cars:
            if street.queueing_cars[0].just_arrived_at_intersection:
                street.queueing_cars[0].just_arrived_at_intersection = False
            else:
                car = street.queueing_cars.pop(0)
                car.cross_intersection()


        self.inner_counter += 1

class Car:
    def __init__(self, *, identifier, total_route):
        self.identifier = identifier
        self.current_street = total_route[0]
        self.total_route = total_route
        self.remaining_route = total_route[1:]
        self.remaining_traveling_time = 0
        self.just_arrived_at_intersection = False

        self.current_street.queueing_cars.append(self)

        self.done = False

    def cross_intersection(self):
        if not self.remaining_route:
            self.done = True
            return

        next_street = self.remaining_route.pop(0)

        self.current_street = next_street
        self.remaining_traveling_time = next_street.T

        next_street.traveling_cars.append(self)

class World:
    def __init__(self, *, config):
        self.config = config

        self.intersections = dict()
        self.streets = dict()
        self.cars = dict()

    def build(self):

        # Generate Intersections
        for id in range(self.config.num_intersections):
            self.intersections[id] = Intersection(id=id)

        # Generate Streets
        for street in self.config.streets:
            street = Street(name=street.name,
                            intersection_start=self.intersections[street.intersection_start],
                            intersection_end=self.intersections[street.intersection_end],
                            T=street.T)
            self.streets[street.name] = street

            self.intersections[street.intersection_start.identifier].out_streets.append(street)
            self.intersections[street.intersection_end.identifier].in_streets.append(street)

            self.intersections[street.intersection_end.identifier].traffic_lights.append(False)

        # Generate Cars
        for id in range(self.config.num_cars):
            total_route = [self.streets[name] for name in self.config.cars[id].total_route]
            car = Car(identifier=id,
                      total_route=total_route)
                    
            self.cars[id] = car

    def step(self, t):
        for street in self.streets.values():
            street.step(t)

        for intersection in self.intersections.values():
            intersection.step(t)

    def print_state(self):
        for intersection_id, intersection in self.intersections.items():
            print('Intersection: ' + str(intersection_id))
            print(' In Streets')
            for i, street in enumerate(intersection.in_streets):
                print('  ', end='')
                if intersection.curr_green == i:
                    print('[Green] ', end='')
                else:
                    print('[Red] ', end='')
                print(street.name + ': ', end='')
                print(', '.join([str(car.identifier) + ' (' + str(car.remaining_traveling_time) + ')' for car in reversed(street.traveling_cars)]), end='')
                print('[', end='')
                print(', '.join([str(car.identifier) for car in reversed(street.queueing_cars)]), end='')
                print(']')
            print(' Out Streets')
            for i, street in enumerate(intersection.out_streets):
                print('  ' + street.name)
            print()