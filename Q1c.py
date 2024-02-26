import numpy as np

def solve_matrix_problem(coefficients, constants):
    """
    Solve a matrix problem given the coefficient matrix and the constant vector.

    Parameters:
    coefficients (ndarray): Coefficient matrix of the system of linear equations.
    constants (ndarray): Constant vector of the system of linear equations.

    Returns:
    ndarray: Array of solutions to the system of linear equations.
    """
    return np.linalg.solve(coefficients, constants)

# Define the coefficient matrix and the constant vector for the first problem
A1 = np.array([[3, 1, -1],
               [1, 4, 1],
               [2, 1, 2]])
b1 = np.array([2, 12, 10])

# Solve the first problem
x1 = solve_matrix_problem(A1, b1)
print("Solution for the first problem:")
for i, x in enumerate(x1, start=1):
    print(f"x{i} = {x}")

# Define the coefficient matrix and the constant vector for the second problem
A2 = np.array([[1, -10, 2, 4],
               [3, 1, 4, 12],
               [9, 2, 3, 4],
               [1, 2, 7, 3]])
b2 = np.array([2, 12, 21, 37])

# Solve the second problem
x2 = solve_matrix_problem(A2, b2)
print("\nSolution for the second problem:")
for i, x in enumerate(x2, start=1):
    print(f"x{i} = {x}")