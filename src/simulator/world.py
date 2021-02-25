class Street:
    def __init__(self, *, name, intersection_start, intersection_end, T):
        self.name = name
        self.intersection_start = intersection_start
        self.intersection_end = intersection_end
        self.T = T
        self.cars = []

    @property
    def has_cars(self):
        return len(self.cars) > 0

    def add_car(self, car):
        self.cars.append(car)

    def remove_car(self, car):
        self.cars.remove(car)

    def get_car_position(self, car):
        self.cars.index(car)

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

    def move(self):
        if self.current_street.get_car_position(self) == 0:


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
