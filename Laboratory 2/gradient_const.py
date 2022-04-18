from methods import Method
import math


class GradientConst(Method):
    def __init__(self, func, eps, n, _lambda, x0: list):
        super().__init__(func, eps, n, _lambda, x0)
        self.answer_point = 0
        self.answer = 0

    def iteration(self, x0):
        gradx0 = self.calculate_gradient(x0)
        _lambda = self.lambda_static()

        s = _lambda * gradx0
        x = lists_sub(x0, s)
        self.iterations += 1
        return x, s

    def run(self):
        x0 = self.x0
        x, s = self.iteration(x0)

        while vector_mod(s) >= self.eps:
            self.segments.append([x0, x])
            x0 = x
            x, s = self.iteration(x0)
            print(vector_mod(lists_sub(x0, x)), self.iterations)
        pre_result = lists_sum(x, x0)
        self.answer = self.func(multiply_list(pre_result, 1 / 2))
        self.answer_point = multiply_list(pre_result, 1 / 2)


def vector_mod(x: list):
    return math.sqrt(sum([i ** 2 for i in x]))


def lists_sub(a: list, b: list):
    if len(a) != len(b):
        raise ArithmeticError("Lists must be same length")
    return [a[i] - b[i] for i in range(len(a))]


def lists_sum(a: list, b: list):
    if len(a) != len(b):
        raise ArithmeticError("Lists must be same length")
    return [a[i] + b[i] for i in range(len(a))]


def multiply_list(a: list, x):
    return [i * x for i in a]
