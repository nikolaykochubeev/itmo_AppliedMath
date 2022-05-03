import math
from functools import reduce
from random import random

import numpy as np


class Line:
    def __init__(self, n: int):
        self.n = n
        self.data = {}

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.tolist()[index]
        if index >= self.n:
            raise IndexError(f"Index {index} out of range")
        return self.data.get(index, 0)

    def __setitem__(self, index: int, value: float):
        if index >= self.n:
            raise IndexError(f"Index {index} out of range")
        if value != 0:
            self.data[index] = value
        elif index in self.data:
            del self.data[index]

    def __len__(self):
        return self.n

    def __str__(self):
        return str([self[i] for i in range(self.n)])

    def tolist(self):
        return [self[i] for i in range(self.n)]

    def toarray(self):
        return np.array(self.tolist())


class Matrix:
    def __init__(self, n: int):
        self.n = n
        self.lines = [Line(n) for _ in range(n)]

    @staticmethod
    def from_data(data: list[list[float]] = None):
        n = len(data)
        m = Matrix(n)
        for i in range(n):
            for j in range(n):
                m[i][j] = data[i][j]
        return m

    @staticmethod
    def random(n: int):
        m = Matrix(n)
        for i in range(n):
            for j in range(n):
                m[i][j] = int(random() * 9 + 1)
        return m

    @staticmethod
    def sparse_random(n: int, density: float = 0.1):
        m = Matrix(n)
        for i in range(n):
            for j in range(n):
                if random() < density:
                    m[i][j] = int(random() * 9 + 1)
        return m

    @staticmethod
    def identity(n: int):
        m = Matrix(n)
        for i in range(n):
            m[i][i] = 1
        return m

    def __getitem__(self, index):
        return self.lines[index]

    def __setitem__(self, key: int, line):
        if len(line) != self.n:
            raise IndexError("Index out of range")
        if isinstance(line, Line):
            self.lines[key] = line
            return
        for value in enumerate(line):
            self[key][value[0]] = value[1]

    def __len__(self):
        return self.n

    def __mul__(self, n: float):
        m = Matrix(self.n)
        for i in range(self.n):
            for j in range(self.n):
                m[i][j] = self[i][j] * n
        return m

    def __rmul__(self, n: float):
        return self.__mul__(n)

    def __matmul__(self, other):
        result = Matrix(self.n)
        for i in range(self.n):
            for j in range(self.n):
                result[i][j] = sum(map(lambda r: self[i][r] * other[r][j], range(self.n)))
        return result

    def __str__(self) -> str:
        return "[" + \
               reduce(lambda prev, line: prev + str(line) + ",\n ", self.lines, "").removesuffix(",\n ") + \
               "]"

    def wolframstr(self) -> str:
        return str(self).replace("\n", " ").replace("  ", " ").replace("[", "{").replace("]", "}")

    def actualstr(self) -> str:
        return "[" + ",\n".join(map(lambda line: "[" + str(line.data) + "]", self.lines)) + "]"

    def tolist(self):
        return list(map(lambda i: i.tolist(), self.lines))

    def toarray(self):
        return np.array(list(map(lambda i: i.toarray(), self.lines)))

    def l(self):
        return LMatrix(self)

    def u(self):
        return UMatrix(self)

    def lu_decomposition(self):
        n = self.n
        l = Matrix(n)
        u = Matrix(n)
        for j in range(n):
            u[j][j] = 1
            for i in range(j, n):
                alpha = float(self[i][j])
                for k in range(j):
                    alpha -= l[i][k] * u[k][j]
                l[i][j] = alpha
            for i in range(j + 1, n):
                temp_u = float(self[j][i])
                for k in range(j):
                    temp_u -= l[j][k] * u[k][i]
                if int(l[j][j]) == 0:
                    l[j][j] = math.e - 40
                u[j][i] = temp_u / l[j][j]
        return l, u

    def det(self) -> float:
        return np.linalg.det(self.tolist())

    def transpose(self):
        result = Matrix(self.n)
        for i in range(self.n):
            for j in range(self.n):
                result[j][i] = self[i][j]
        return result

    def inverse(self):
        return self.u().inverse() @ self.l().inverse()

    def valid_inverse(self):
        return np.linalg.inv(self.toarray())


class LMatrix(Matrix):
    def __init__(self, base: Matrix):
        super().__init__(base.n)
        l, u = base.lu_decomposition()
        for i in range(base.n):
            for j in range(base.n):
                self[i][j] = l[i][j]

    def det(self) -> float:
        diagonal = map(lambda num_line: num_line[1][num_line[0]], enumerate(self.lines))
        return reduce(lambda v1, v2: v1 * v2, diagonal, 1)

    def inverse(self) -> Matrix:
        return np.linalg.solve(self, Matrix.identity(self.n))


class UMatrix(Matrix):
    def __init__(self, base: Matrix):
        super().__init__(base.n)
        l, u = base.lu_decomposition()
        for i in range(base.n):
            for j in range(base.n):
                self[i][j] = u[i][j]

    def det(self) -> float:
        return 1

    def inverse(self):
        return np.linalg.solve(self, Matrix.identity(self.n))
