from math import *

import numpy as np


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

    def calculate_brent_method(self):
        a = self.a
        b = self.b
        epsilon = self.epsilon
        iteration = 0
        K = (3 - np.sqrt(5)) / 2
        x = (a + b) / 2
        w = x
        v = x
        f_x = self.function(x)
        f_w = f_x
        f_v = f_x
        # длины текущего и предыдущего шага
        d = b - a
        e = d
        while b - a >= epsilon:
            print(f"Left border: {a}, Right border: {b}")
            iteration += 1
            g = e
            e = d
            parabola_fit = False
            # критерий остановки
            if abs(x - (a + b) / 2) + (b - a) / 2 <= 2 * epsilon:
                break
            if (x != w != w != v) and (f_x != f_w != f_v):
                f_1, f_2, f_3 = self.function(x), self.function(w), self.function(v)
                # Параболическая аппроксимация, находим u – минимум параболы
                u = w - ((w - x) ** 2 * (f_2 - f_3) - (w - v) ** 2 * (f_2 - f_1)) / (
                        2 * ((w - x) * (f_2 - f_3) - (w - v) * (f_2 - f_1)))
                if u >= a + epsilon and u <= b - epsilon and abs(u - x) < g / 2:
                    d = abs(u - x)
                    parabola_fit = True
                    if (u - a < 2 * epsilon) or (b - u) < 2 * epsilon:
                        u = x - np.sign(x - (a + b) / 2) * epsilon
            if not parabola_fit:
                # Золотое сечение [x, c];
                if x < (b + a) / 2:
                    u = x + K * (b - x)
                    d = b - x
                # Золотое сечение [a, x]
                else:
                    u = x - K * (x - a)
                    d = x - a
                if abs(u - x) < epsilon:
                    # Задаём минимальную близость между u и x
                    u = x + np.sign(u - x) * epsilon
            f_u = self.function(u)
            d = abs(u - x)
            if f_u <= f_x:
                if u >= x:
                    a = x
                else:
                    b = x
                v = w
                w = x
                x = u
                f_v = f_w
                f_w = f_x
                f_x = f_u
            else:
                if u >= x:
                    b = u
                else:
                    a = u
                if f_u <= f_w or w == x:
                    v = w
                    w = u
                    f_v = f_w
                    f_w = f_u
                elif f_u <= f_v or v == x or v == w:
                    v = u
                    f_v = f_u
        print(f"Min(x): x = {(a + b) / 2}, Min(y): y = {self.function((a + b) / 2)}")
        print(f"Number of iterations - {iteration}")


optimization = Optimization(lambda x: sin(x) * x ** 2, -3, -2, 1e-5)
optimization.calculate_dichotomy()
optimization.calculate_brent_method()

# метод Брента комбинирует метод золотого сечения и парабол
# на каждой итерации отслеживается значение в 6 точках
# a и c - текущий интервал поиска решений
# x точка соответствующая наименьшему значению функции (среднее)
# w - второму снизу значению функции
# v - предыдущее значение w
# апроксимирующая парабола строится с помощью трех наилучших точек x, w, v
# u - min апроксимирующий параболы принимается в качестве следующей точки оптимизационного процесса если:
# 1) u попало внутрь [a, c] и ([a, u] > e), ([u, c] > e) и расстояние от u до x оно не больше половины длины предыдущего шага
# Если точка не подходит, то следующая точка будет находится с помощью золотого сечения большего из интервалов max{[a, x][x, c]}