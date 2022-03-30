from math import *


class Optimization:
    def __init__(self, function, a, b, epsilon):
        self.function = function
        self.a = a
        self.b = b
        self.epsilon = epsilon

    def calculate_dichotomy(self):
        n = 0
        a = self.a
        b = self.b
        f1 = self.function(a)
        f2 = self.function(b)
        while (b - a) / 2 > self.epsilon:
            x1 = (a + b) / 2 - self.epsilon / 2
            x2 = (a + b) / 2 + self.epsilon / 2
            if f1 < f2:
                b = x2
                f2 = self.function(b)
            elif f1 > f2:
                a = x1
                f1 = self.function(a)
            else:
                a = x1
                b = x2
                f1 = self.function(a)
                f2 = self.function(b)
            n = n + 1
        x = (a + b) / 2
        print('x = ', x)
        print('f(x) = ', self.function(x))
        print('n = ', n)


optimization = Optimization(lambda x: sin(x) * x ** 2, -3, -2, 1e-5)
optimization.calculate_dichotomy()
