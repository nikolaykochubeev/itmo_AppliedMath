from numdifftools import Gradient
import matplotlib.pyplot as plt
import numpy as np


def calculate_gradient(func, x1, x2):
    return Gradient(func)(np.asarray([x1, x2]))


def gradient_x(func, x1, x2):
    return calculate_gradient(lambda x: func(x[0], x[1]), x1, x2)[0]


def gradient_y(func, x1, x2):
    return calculate_gradient(lambda x: func(x[0], x[1]), x1, x2)[1]


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


def calculate_golden_ratio(func, x, y, grad_x, grad_y, a, b, epsilon):
    one_dim_func = lambda param: func(x - grad_x * param, y - grad_y * param)

    x1 = b - (b - a) * 0.61803398874
    x2 = a + (b - a) * 0.61803398874
    f1 = one_dim_func(x1)
    f2 = one_dim_func(x2)
    while (x2 - x1) / 2 > epsilon:
        if f1 < f2:
            b = x2
            x2 = x1
            x1 = b - (b - a) * 0.61803398874
            f2 = f1
            f1 = one_dim_func(x1)
        else:
            a = x1
            x1 = x2
            x2 = a + (b - a) * 0.61803398874
            f1 = f2
            f2 = one_dim_func(x2)
    return (a + b) / 2


def calculate_fibonacci(func, x, y, grad_x, grad_y, a, b, epsilon):
    one_dim_func = lambda param: func(x - grad_x * param, y - grad_y * param)

    n = 0
    while get_fibonacci_sequence(n)[-1] <= (b - a) / epsilon:
        n += 1
    sequence = get_fibonacci_sequence(n)

    x1 = a + (b - a) * (sequence[n - 1] / sequence[n + 1])
    x2 = a + (b - a) * (sequence[n] / sequence[n + 1])
    f1 = one_dim_func(x1)
    f2 = one_dim_func(x2)
    while n > 0:
        n -= 1
        if f1 < f2:
            b = x2
            x2 = x1
            x1 = a + (b - x2)
            f2 = f1
            f1 = one_dim_func(x1)
        else:
            a = x1
            x1 = x2
            x2 = b - (x1 - a)
            f1 = f2
            f2 = one_dim_func(x2)
    return (x1 + x2) / 2


def draw(a, b, func, points_x, points_y, name):
    fig, ax = plt.subplots()
    x, y = np.mgrid[a:b:100j, a:b:100j]
    ax.set_title(name)
    ax.contour(x, y, func(x, y), levels=100, colors='#0039A6')
    for i in range(len(points_x)):
        ax.scatter(points_x[i], points_y[i], c='#D52B1E')
    ax.plot([points_x[i] for i in range(len(points_x))], [points_y[i] for i in range(len(points_y))], c='#D52B1E')
    plt.show()


def gradient_descent(func, is_constant, x_start, y_start, a, b, epsilon):
    funny_symbol_e = 0.25
    x, y = x_start, y_start
    x_next, y_next = x, y
    points_x = np.asarray([]).astype(float)
    points_y = np.asarray([]).astype(float)
    step = epsilon
    k = 0
    if not is_constant:
        step *= 10
    while True:
        k += 1

        points_x = np.concatenate((points_x, [x_next]))
        points_y = np.concatenate((points_y, [y_next]))
        grad_x = gradient_x(func, x, y)
        grad_y = gradient_y(func, x, y)

        grad_norm_squared = np.linalg.norm([grad_x, grad_y]) ** 2

        x_next = x - step * grad_x
        y_next = y - step * grad_y

        if not is_constant:
            while func(x_next, y_next) > func(x, y) - funny_symbol_e * step * grad_norm_squared:
                step *= funny_symbol_e

        if abs(x - x_next) < epsilon and abs(y - y_next) < epsilon and grad_x < epsilon and grad_y < epsilon and abs(
                func(x, y) - func(x_next, y_next)) < epsilon:
            break

        if k > 2000:
            break

        x = x_next
        y = y_next

    print(k - 1)
    draw(a, b, func, points_x, points_y, 'метод градиентного спуска')
    return func(x, y)


def steepest_descent(func, optimizer, x_start, y_start, a, b, epsilon):
    x, y = x_start, y_start
    x_next, y_next = x, y
    points_x = np.asarray([]).astype(float)
    points_y = np.asarray([]).astype(float)
    k = 0
    while True:
        k += 1

        points_x = np.concatenate((points_x, [x_next]))
        points_y = np.concatenate((points_y, [y_next]))
        grad_x = gradient_x(func, x, y)
        grad_y = gradient_y(func, x, y)

        step = optimizer(func, x, y, grad_x, grad_y, a, b, epsilon)

        x_next = x - step * grad_x
        y_next = y - step * grad_y

        if abs(x - x_next) < epsilon and abs(y - y_next) < epsilon and grad_x < epsilon and grad_y < epsilon and abs(
                func(x, y) - func(x_next, y_next)) < epsilon:
            break

        if k > 2000:
            break

        x = x_next
        y = y_next

    print(k - 1)
    draw(a, b, func, points_x, points_y, 'метод наискорейшего спуска')
    return func(x, y)


def conjugate_gradient(func, optimizer, x_start, y_start, a, b, epsilon):
    prev_basis_x = 0
    prev_basis_y = 0
    x, y = x_start, y_start
    x_next, y_next = x, y
    grad_norm, prev_grad_norm = 0, 0
    points_x = np.asarray([]).astype(float)
    points_y = np.asarray([]).astype(float)
    k = 0
    while True:
        k += 1

        points_x = np.concatenate((points_x, [x_next]))
        points_y = np.concatenate((points_y, [y_next]))

        if prev_grad_norm == 0 or k % 3 == 0:
            beta = 0
            prev_basis_x = -gradient_x(func, x, y)
            prev_basis_y = -gradient_y(func, x, y)
        else:
            beta = grad_norm ** 2 / prev_grad_norm ** 2

        grad_x = gradient_x(func, x, y)
        grad_y = gradient_y(func, x, y)

        grad_norm = np.linalg.norm([grad_x, grad_y])
        step = optimizer(func, x, y, grad_x, grad_y, a, b, epsilon)

        x_next = x + step * prev_basis_x
        y_next = y + step * prev_basis_y

        basis_x = -gradient_x(func, x_next, y_next) + beta * prev_basis_x
        basis_y = -gradient_y(func, x_next, y_next) + beta * prev_basis_y

        if abs(x - x_next) / 2 < epsilon and abs(y - y_next) / 2 < epsilon or abs(grad_x) < epsilon and abs(
                grad_y) < epsilon or abs(func(x_next, y_next) - func(x, y)) / 2 < epsilon:
            break

        if k > 2000:
            break

        x, y = x_next, y_next
        prev_basis_x, prev_basis_y = basis_x, basis_y
        prev_grad_norm = grad_norm

    print(k - 1)
    draw(a, b, func, points_x, points_y, 'метод сопряжённых градиентов')
    return func(x, y)


def my_r2_func(x, y):
    return -(np.exp((-(((x - 4) ** 2 + (y - 4) ** 2) ** 2)) / 1000) + np.exp(
        (-(((x + 4) ** 2 + (y + 4) ** 2) ** 2)) / 1000) + 0.15 * np.exp(
        -(((x + 4) ** 2 + (y + 4) ** 2) ** 2)) + 0.15 * np.exp(-(((x - 4) ** 2 + (y - 4) ** 2) ** 2)))
    # return (x1 ** 2 + x2 - 11) ** 2 + (x1 + x2 ** 2 - 7) ** 2
    # return ((np.sin(3. * np.pi * x1)) ** 2.)\
    #        + ((x1 - 1.) ** 2.) * (1. + (np.sin(3. * np.pi * x2)) ** 2.)\
    #        + ((x2 - 1.) ** 2.) * (1. + (np.sin(2. * np.pi * x2)) ** 2.)
    # return 0.26 * (x1 ** 2 + x2 ** 2) - 0.48 * x1 * x2
    # return (x1 + 2 * x2 - 7) ** 2 + (2 * x1 + x2 - 5) ** 2
    # return 5 * x1 ** 2 + 73 * x2 ** 2


def test(func, start_x, start_y, a, b, epsilon):
    # print(gradient_descent(func, True, start_x, start_y, a, b, epsilon))
    # print(gradient_descent(func, False, start_x, start_y, a, b, epsilon))

    print(steepest_descent(func, calculate_golden_ratio, start_x, start_y, a, b, epsilon))
    print(steepest_descent(func, calculate_fibonacci, start_x, start_y, a, b, epsilon))

    print(conjugate_gradient(func, calculate_golden_ratio, start_x, start_y, a, b, epsilon))
    print(conjugate_gradient(func, calculate_fibonacci, start_x, start_y, a, b, epsilon))


if __name__ == '__main__':
    test(my_r2_func, -3.001, 3, -10, 10, 0.001)
