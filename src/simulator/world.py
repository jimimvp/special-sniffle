class Street:
    def __init__(self, *, name, intersection_in, intersection_out, T):
        self.name = name
        self.intersection_in = intersection_in
        self.intersection_out = intersection_out
        self.T = T
        self.cars = []

    @property
    def has_cars(self):
        return len(self.cars) > 0

    def add_car(self, car):
        self.cars.append(car)

    def remove_car(self, car):
        self.cars.remove(car)

class Intersection:
    def __init__(self, *, id):
        self.identifier = id
        self.in_streets = []
        self.out_streets = []

class Car:
    def __init__(self):
        pass

class World:
    def __init__(self, *, config):
        self.config = config
        self.build()

        self.streets = dict()
        self.intersections = dict()

    def build(self):

        for id in self.config.num_intersections:
            self.intersections[id] = Intersection(id=id)

        for street in self.config.streets:
            street = Street(name=street.name,
                            intersection_in=self.intersections[street.intersection_in],
                            intersection_out=self.intersections[street.intersection_out],
                            T=street.T)
            self.streets[street.name] = street
