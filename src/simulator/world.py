class Street:
    def __init__(self, *, name, intersection_start, intersection_end, T):
        self.name = name
        self.intersection_start = intersection_start
        self.intersection_end = intersection_end
        self.T = T
        self.cars = []
        self.traveling_cars = []
        self.queueing_cars = [] # FIFO
    
    def step(self):
        for car in self.traveling_cars:
            self.remaining_traveling_time -= 1
            if car.remaining_traveling_time == 0:
                self.traveling_cars.remove(car)
                self.queueing_cars.append(car)

        if self.intersection_end.traffic_light[self.name] is True:
            car = self.queueing_cars.pop(0)
            car.cross_intersection()
            
class Intersection:
    def __init__(self, *, id):
        self.identifier = id
        self.in_streets = []
        self.out_streets = []

class Car:
    def __init__(self, *, identifier, total_route):
        self.identifier = identifier
        self.current_street = []
        self.total_route = total_route
        self.remaining_route = []
        self.remaining_traveling_time = 1

    def cross_intersection(self):
        next_street = self.remaining_route.pop(0)

        self.current_street = next_street
        self.remaining_traveling_time = next_street.T

        next_street.traveling_cars.append(self)

class World:
    def __init__(self, *, config):
        self.config = config
        self.build()

        self.intersections = dict()
        self.streets = dict()
        self.cars = dict()

    def build(self):

        # Generate Intersections
        for id in self.config.num_intersections:
            self.intersections[id] = Intersection(id=id)

        # Generate Streets
        for street in self.config.streets:
            street = Street(name=street.name,
                            intersection_start=self.intersections[street.intersection_start],
                            intersection_end=self.intersections[street.intersection_end],
                            T=street.T)
            self.streets[street.name] = street

            self.intersections[street.intersection_start].out_streets.append(street)
            self.intersections[street.intersection_end].in_streets.append(street)

        # Generate Cars
        for id in self.config.num_cars:
            car = Car(identifier=id,
                      total_route=self.config.cars[id].plan)
                    
            self.cars[id] = car
