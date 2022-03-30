from math import sin


class Optimization:
    def __init__(self, function):
        self.function = function

    def calculate_dichotomy(self, a, b):
        epsilon = 1e-5
        while b - a > epsilon:
            x1 = (a + b) / 2 - epsilon / 3
            x2 = (a + b) / 2 + epsilon / 3
            if self.function(x1) < self.function(x2):
                b = x2
            elif self.function(x1) > self.function(x2):
                a = x1
            else:
                a = x1
                b = x2
        print((a + b) / 2)


optimization = Optimization(lambda x: sin(x) * x ** 2)
optimization.calculate_dichotomy(-3, -2)
