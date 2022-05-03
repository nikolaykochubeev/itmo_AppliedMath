import numpy as np

from matrix_generator import *
from gauss_seidel import gauss_seidel_solve
from matrix import Matrix
from lu_decomposition import *
import matplotlib.pyplot as plt


def diagonal_dominant_deltas(solver, n=5, amount=10):
    matrices = diagonal_dominant(n)
    answers = [np.dot(matrix, list(range(1, n+1))) for matrix in matrices]
    cond_numbers = [np.linalg.cond(matrix) for matrix in matrices]
    deltas = []
    for i in range(amount):
        deltas.append(find_delta(solver(matrices[i], answers[i]), np.linalg.solve(matrices[i], answers[i])))
    return deltas, cond_numbers


def hilbert_deltas(solver, dimensions=list(range(3, 13))):
    matrices = [hilbert(i) for i in dimensions]
    answers = [np.dot(matrices[i], list(range(1, dimensions[i] + 1))) for i in range(len(matrices))]
    deltas = []
    for i in range(len(dimensions)):
        deltas.append(find_delta(solver(matrices[i], answers[i]), np.linalg.solve(matrices[i], answers[i])))
    return deltas


def find_delta(v1, v2):
    result = []
    for i in range(len(v1)):
        result.append(abs(v1[i] - v2[i]))
    return result


def create_seidel_table():
    deltas, cond_numbers = diagonal_dominant_deltas(gauss_seidel_solve)

    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.table(
        cellText=deltas,
        colLabels=[f"x{i}" for i in range(1, 6)],
        colColours=["steelblue"] * len(cond_numbers),
        cellLoc='center',
        loc='upper left'
    )
    ax.set_title('Precision depending on cond number, seidel method',
                 fontweight="bold")
    # plt.show()
    plt.savefig("seidel_precision.png", dpi=300)


def create_lu_table():
    deltas, cond_numbers = diagonal_dominant_deltas(linear_solve_without, n=5, amount=5)

    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.table(
        cellText=deltas,
        colLabels=[f"x{i}" for i in range(1, 6)],
        colColours=["steelblue"] * len(cond_numbers),
        cellLoc='center',
        loc='upper left'
    )
    ax.set_title('Precision depending on cond number, LU decomposition method',
                 fontweight="bold")
    # plt.show()
    plt.savefig("lu_precision.png", dpi=300)


def create_hilbert_table_with_seidel():
    deltas = hilbert_deltas(linear_solve_without, dimensions=[10, 50, 100])

    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.table(
        cellText=[[sum(delta)] for delta in deltas],
        rowLabels=[10, 50, 100],
        colLabels=["delta X"],
        rowColours=["steelblue"] * len(deltas),
        colColours=["steelblue"],
        cellLoc='center',
        loc='upper left'
    )
    ax.set_title('Precision depending on cond number, hilbert matrices\nSeidel method',
                 fontweight="bold")
    # plt.show()
    plt.savefig("hilbert_precision_seidel.png")


def create_hilbert_table_with_lu():
    deltas = hilbert_deltas(linear_solve_without, dimensions=[10, 50, 100])

    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.table(
        cellText=[[sum(delta)] for delta in deltas],
        rowLabels=[10, 50, 100],
        colLabels=["delta X"],
        rowColours=["steelblue"] * len(deltas),
        colColours=["steelblue"],
        cellLoc='center',
        loc='upper left'
    )
    ax.set_title('Precision depending on cond number, hilbert matrices\nLU decomposition',
                 fontweight="bold")
    # plt.show()
    plt.savefig("hilbert_precision_lu.png")


def compare_methods_by_iterations():
    dimensions = [3, 4, 5, 6, 7, 8, 9, 10, 50, 100, 200]
    seidel_iters = []
    lu_iters = []
    for dim in dimensions:
        matrix = diagonal_dominant(dim, amount=1)[0]
        b = list(range(1, dim + 1))
        x, iters1 = gauss_seidel_solve(matrix, b)
        seidel_iters.append(iters1)
        x, iters2 = linear_solve_without(matrix, b)
        lu_iters.append(iters2)
    fig, ax = plt.subplots()
    ax.plot(dimensions, seidel_iters)
    plt.show()
    fig.savefig(f"plots/iter_seidel")
    ax.plot(dimensions, lu_iters)
    fig.savefig(f"plots/iter_lu")


compare_methods_by_iterations()