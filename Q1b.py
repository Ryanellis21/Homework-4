from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt

def equation1(x):
    """
    Equation 1: x - 3 * cos(x) = 0
    """
    return x - 3 * np.cos(x)

def equation2(x):
    """
    Equation 2: cos(2 * x) * x^3 = 0
    """
    return np.cos(2 * x) * x**3

# Find the roots of the equations
root1 = fsolve(equation1, 0)  # Initial guess = 0
root2 = fsolve(equation2, 0)  # Initial guess = 0

# Check if the functions intersect
if np.isclose(root1, root2):
    intersection_point = root1
    intersection_exists = True
else:
    intersection_exists = False

# Plot the functions
x = np.linspace(-10, 10, 1000)
y1 = equation1(x)
y2 = equation2(x)

plt.plot(x, y1, label='x - 3 * cos(x)')
plt.plot(x, y2, label='cos(2 * x) * x^3')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Intersection of Functions')
plt.legend()
plt.grid(True)
plt.show()

# Print the results
print("Root 1:", root1)
print("Root 2:", root2)

if intersection_exists:
    print("The functions intersect at x =", intersection_point)
else:
    print("The functions do not intersect.")