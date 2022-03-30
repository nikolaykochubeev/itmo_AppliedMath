from math import *


class Optimization:
    def __init__(self, function, a, b, epsilon):
        self.function = function
        self.a = a
        self.b = b
        self.epsilon = epsilon

    def calculate_dichotomy(self):
        n = 0
        result = 0
        a = self.a
        b = self.b
        fa = self.function(a)
        fb = self.function(b)
        while (b - a) / 2 > self.epsilon:
            result = (a + b) / 2
            if fa <= fb:
                b = result
                fb = self.function(b)
            else:
                a = result
                fa = self.function(a)
            n = n + 1
        print(result, n)


optimization = Optimization(lambda x: sin(x) * x ** 2, -3, -2, 1e-5)
optimization.calculate_dichotomy()
