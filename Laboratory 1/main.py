from math import sin


def f(x):
    return sin(x) * x ** 2


def calculate_dichotomy(a, b):
    epsilon = 1e-5
    while b - a > epsilon:
        x1 = (a + b) / 2 - epsilon / 3
        x2 = (a + b) / 2 + epsilon / 3
        if f(x1) < f(x2):
            b = x2
        elif f(x1) > f(x2):
            a = x1
        else:
            a = x1
            b = x2
    print((a + b) / 2)


calculate_dichotomy(-3, -2)
