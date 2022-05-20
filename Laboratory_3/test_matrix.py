from LU_Solution import solve
from systems_solver import solve_systems, generate_gilbert_matrix, generate_diagonal_matrix
from method_Zeldel import solution
from print_Matrix import print_matrix

solve_systems(generate_gilbert_matrix, solution, 14)
solve_systems(generate_diagonal_matrix, solution, 14)
