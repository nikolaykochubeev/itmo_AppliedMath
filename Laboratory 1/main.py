from math import sin


class Optimization:
    def __init__(self, function, a, b, epsilon):
        self.function = function
        self.a = a
        self.b = b
        self.epsilon = epsilon
        self.ratio = 0.38196601125
        self.u = lambda x1, x2, x3, f1, f2, f3: x2 - ((x2 - x1) ** 2 * (f2 - f3) - (x2 - x3) ** 2 * (f2 - f1)) / (
                2 * ((x2 - x1) * (f2 - f3) - (x2 - x3) * (f2 - f1)))

    def calculate_dichotomy(self):
        a = self.a
        b = self.b
        n = 0
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
            n = n + 1
        x = (a + b) / 2
        print('x = ', x)
        print('f(x) = ', self.function(x))
        print('n = ', n)

    def calculate_golden_ratio(self):
        n = 0
        a = self.a
        b = self.b
        x1 = a + (b - a) * self.ratio
        x2 = b - (b - a) * self.ratio
        f1 = self.function(x1)
        f2 = self.function(x2)
        while (x2 - x1) / 2 > self.epsilon:
            if f1 < f2:
                b = x2
                x2 = x1
                x1 = a + (b - a) * self.ratio
                f2 = f1
                f1 = self.function(x1)
            else:
                a = x1
                x1 = x2
                x2 = b - (b - a) * self.ratio
                f1 = f2
                f2 = self.function(x2)
            n = n + 1
        x = (a + b) / 2
        print('x = ', x)
        print('f(x) = ', self.function(x))
        print('n = ', n)

    def calculate_parabola(self):
        x1 = self.a
        x3 = self.b
        x2 = (x3 - x1) / 2
        x_i = 0
        n = 0
        f1 = self.function(x1)
        f2 = self.function(x2)
        f3 = self.function(x3)
        # u = self.u(x1, x2, x3, f1, f2, f3)
        while True:
            u = self.u(x1, x2, x3, f1, f2, f3)
            fu = self.function(u)
            if fu <= f2:
                if u >= x2:
                    x1 = x2
                    f1 = f2
                else:
                    x3 = x2
                    f3 = f2
                x2 = u
                f2 = fu
            else:
                if x2 < u:
                    x3 = u
                    f3 = fu
                else:
                    x1 = u
                    f1 = fu
            if abs(x_i - u) < self.epsilon:
                break
            x_i = u
            n += 1

        print('x = ', u)
        print('f(x) = ', self.function(u))
        print('n = ', n)


optimization = Optimization(lambda x: sin(x) * x ** 2, -3, -2, 1e-5)
optimization.calculate_dichotomy()
optimization.calculate_golden_ratio()
optimization.calculate_parabola()
