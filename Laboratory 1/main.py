from math import sin

import numpy as np
from matplotlib import pyplot as plt


class Optimization:
    def __init__(self, function, a, b, epsilon):
        self.function = function
        self.a = a
        self.b = b
        self.epsilon = epsilon
        self.ratio = 0.61803398874
        self.u = lambda x1, x2, x3, f1, f2, f3: x2 - ((x2 - x1) ** 2 * (f2 - f3) - (x2 - x3) ** 2 * (f2 - f1)) / (
                2 * ((x2 - x1) * (f2 - f3) - (x2 - x3) * (f2 - f1))) \
            if (2 * ((x2 - x1) * (f2 - f3) - (x2 - x3) * (f2 - f1))) != 0 \
            else None

    def plot_function(self):
        vectorize_function = np.vectorize(self.function)
        array = np.linspace(self.a, self.b, 100)
        plt.title("sin(x) * x ** 2")
        plt.xlabel("X axis")
        plt.ylabel("Y axis")
        plt.grid()
        plt.plot(array, vectorize_function(array))
        plt.show()

    @staticmethod
    def get_fibonacci_sequence(n):
        n1 = 0
        n2 = 1
        sequence = []
        iterations = 0
        while iterations < n + 2:
            sequence.append(n1)
            nth = n1 + n2
            n1 = n2
            n2 = nth
            iterations += 1
        return sequence

    def calculate_dichotomy(self):
        print('dichotomy method')
        a = self.a
        b = self.b
        prev_length = b - a
        iterations = 0
        while (b - a) / 2 > self.epsilon:
            x1 = (a + b) / 2 - self.epsilon / 2
            x2 = (a + b) / 2 + self.epsilon / 2
            if self.function(a) < self.function(b):
                b = x2
            elif self.function(a) > self.function(b):
                a = x1
            else:
                a = x1
                b = x2
            iterations += 1
            print(iterations, ': ', b - a, ' ', (b - a) / prev_length)
            prev_length = b - a
        x = (a + b) / 2
        print('x = ', x)
        print('f(x) = ', self.function(x))
        print('iterations = ', iterations, '\n')

    def calculate_golden_ratio(self):
        print('golden ratio method')
        a = self.a
        b = self.b
        prev_length = b - a
        iterations = 0
        x1 = b - (b - a) * self.ratio
        x2 = a + (b - a) * self.ratio
        f1 = self.function(x1)
        f2 = self.function(x2)
        while (x2 - x1) / 2 > self.epsilon:
            if f1 < f2:
                b = x2
                x2 = x1
                x1 = b - (b - a) * self.ratio
                f2 = f1
                f1 = self.function(x1)
            else:
                a = x1
                x1 = x2
                x2 = a + (b - a) * self.ratio
                f1 = f2
                f2 = self.function(x2)
            iterations += 1
            print(iterations, ': ', b - a, ' ', (b - a) / prev_length)
            prev_length = b - a
        x = (a + b) / 2
        print('x = ', x)
        print('f(x) = ', self.function(x))
        print('iterations = ', iterations, '\n')

    def calculate_fibonacci(self, n):
        print('fibonacci method')
        a = self.a
        b = self.b
        prev_length = b - a
        sequence = self.get_fibonacci_sequence(n)
        x1 = a + (b - a) * (sequence[n - 1] / sequence[n + 1])
        x2 = a + (b - a) * (sequence[n] / sequence[n + 1])
        f1 = self.function(x1)
        f2 = self.function(x2)
        iterations = 0
        while n > 0:
            n -= 1
            if f1 < f2:
                b = x2
                x2 = x1
                x1 = a + (b - x2)
                f2 = f1
                f1 = self.function(x1)
            else:
                a = x1
                x1 = x2
                x2 = b - (x1 - a)
                f1 = f2
                f2 = self.function(x2)
            iterations += 1
            print(iterations, ': ', b - a, ' ', (b - a) / prev_length)
            prev_length = b - a
        x = (a + b) / 2
        print('x = ', x)
        print('f(x) = ', self.function(x))
        print('iterations = ', iterations, '\n')

    def calculate_parabola(self):
        print('parabola method')
        x1 = self.a
        x3 = self.b
        x2 = (x3 + x1) / 2
        x_i = 0
        prev_length = x3 - x1
        iterations = 0
        f1 = self.function(x1)
        f2 = self.function(x2)
        f3 = self.function(x3)
        u = self.u(x1, x2, x3, f1, f2, f3)
        fu = self.function(u)
        while abs(x_i - u) >= self.epsilon:
            x_i = u
            iterations += 1
            if fu <= f2:
                if u >= x2:
                    print(iterations, ':  ', u - x2, ' ', (u - x2) / prev_length)
                    prev_length = u - x2
                    x1 = x2
                    f1 = f2
                else:
                    print(iterations, ':  ', x3 - u, ' ', (x3 - u) / prev_length)
                    prev_length = x3 - u
                    x3 = x2
                    f3 = f2
                x2 = u
                f2 = fu
            else:
                if x2 < u:
                    print(iterations, ':  ', u - x2, ' ', (u - x2) / prev_length)
                    prev_length = u - x2
                    x3 = u
                    f3 = fu
                else:
                    print(iterations, ':  ', x3 - u, ' ', (x3 - u) / prev_length)
                    prev_length = x3 - u
                    x1 = u
                    f1 = fu
            u = self.u(x1, x2, x3, f1, f2, f3)
            fu = self.function(u)

        print('x = ', u)
        print('f(x) = ', self.function(u))
        print('iterations = ', iterations, '\n')

    def calculate_brent(self):
        print('brent combined method')
        a = self.a
        b = self.b
        iterations = 0
        x = w = v = a + self.ratio * (b - a)
        d = e = b - a
        prev_length = 1
        fx = self.function(x)
        fw = self.function(x)
        fv = self.function(x)
        while max(abs(x - a), abs(b - x)) >= self.epsilon:
            g = e / 2
            e = d
            u = self.u(x, w, v, fx, fw, fv)
            if u is None:
                if x >= (a + b) / 2:
                    u = x - self.ratio * (x - a)
                    e = x - a
                else:
                    u = x + self.ratio * (b - x)
                    e = b - x

            d = abs(u - x)
            fu = self.function(u)
            if fu > fx:
                if u >= x:
                    b = u
                else:
                    a = u
                if fu <= fw or w == x:
                    fv = fw
                    v = w
                    w = u
                    fw = fu
                else:
                    if fu <= fv or x == v or v == w:
                        v = u
                        fv = fu
            else:
                if u >= x:
                    a = x
                else:
                    b = x

            fv = fw
            fw = fx
            fx = fu
            v = w
            w = x
            x = u
            iterations += 1

            print(iterations, ':  ', b - a, ' ', (b - a) / prev_length)
            prev_length = b - a

        print('x = ', x)
        print('f(x) = ', self.function(x))
        print('iterations = ', iterations, '\n')


optimization = Optimization(lambda x: sin(x) * x ** 2, -3, -1, 1e-5)
optimization.calculate_dichotomy()
optimization.calculate_golden_ratio()
optimization.calculate_fibonacci(25)
optimization.calculate_parabola()
optimization.calculate_brent()
optimization.plot_function()
