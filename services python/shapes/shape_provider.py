import random
from shapes.shapes_data import shapes

class ShapeProvider:
    def __init__(self):
        self.current_type = "yoga"
        self.current_shape = None

    def select_shape_type(self, choice):
        types = {"1": "yoga", "2": "sports", "3": "geometric"}
        self.current_type = types.get(choice, "yoga")

    def get_random_shape(self):
        self.current_shape = random.choice(shapes[self.current_type])
        return self.current_shape
