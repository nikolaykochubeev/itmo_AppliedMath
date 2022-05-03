from matrix import Matrix
from numpy import linalg
from gauss_seidel import *

if __name__ == '__main__':
    m = Matrix.random(5)
    # m = Matrix.from_data([
    #     [8, 1, 1],
    #     [1, 4, 9],
    #     [8, 9, 6]])
    # m = Matrix.from_data([
    #     [4, 0, 1],
    #     [0, 2, 1],
    #     [4, 2, 3]])
    # m = Matrix.from_data([
    #     [40, 5, 1],
    #     [2, 60, 2],
    #     [1, 7, 10]])

    print("M:")
    print(m)
    print("M^(-1):")
    print(m.inverse())
    print("Valid inverse:")
    print(m.valid_inverse())

    print("L:")
    print(m.l())
    print("L^(-1):")
    print(m.l().inverse())

    print("U:")
    print(m.u())
    print("U^(-1):")
    print(m.u().inverse())

    print("L @ U:")
    print(m.l() @ m.u())
    print("Valid L @ U:")
    print(m.l().toarray() @ m.u().toarray())

    A = Matrix.from_data([[4, -1, -1],
                          [-2, 6, 1],
                          [-1, 1, 7]])
    b = [3, 9, -6]

    print("Gauss-Seidel method:")
    print("AX=b")
    print("A:", A, sep="\n")
    print("b =", b)
    print("X =", gauss_seidel_solve(A, b))
    print("X (numpy) =", linalg.solve(A, b))

    m = Matrix(1000)
    m[1][50] = 10
    m[20][100] = 30
    print("1000x1000 matrix actual representation (without printing empty lines):")
    print(m.actualstr().replace("[{}],\n", ""))

    A2 = Matrix.from_data([[10., -1., 2., 0.],
                           [-1., 11., -1., 3.],
                           [2., -1., 10., -1.],
                           [0., 3., -1., 8.]])
    b2 = np.array([6.0, 25.0, -11.0, 15.0])
    print("numpy solve:")
    print(linalg.solve(A2, b2))
    print(gauss_seidel_solve(A2, b2))

    m = Matrix.sparse_random(1000)
    print(m)
    print(m.l())
    print(m.u())
    print(m.inverse())
