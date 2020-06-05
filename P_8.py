"""
Author: Pruthvi Suryadevara
Email:  pruthvi.suryadevara@tifr.res.in
Solving boundary value problems using solve_bvp

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as inte
import csv
import os


def f1(x, y):     # Defining second order DE as a pair of first order DE
    return(np.vstack([y[1], 4*(y[0] - x)]))


def bc1(ya, yb):  # Defining the boundary conditions
    return(np.array([ya[0], yb[0]-2]))


t = np.linspace(0, 1, 100)
y_a = np.zeros([2, len(t)])       # Initializing the value of y to zeros
y = inte.solve_bvp(f1, bc1, t, y_a)  # solving using solve_bvp
t_p = np.linspace(0, 1, 200)          # Defining mesh of t for plotting
if y.success:   # Testing for sucess of solution
    print(" The solution is converged to desired accuracy")
else:
    print(" The solution has not converged to desired accuracy")

# Plotting the numerical and actual solutions
y_num = y.sol(t_p)[0]
plt.plot(t_p, y_num, 'r', linewidth=4, label="Numerical Solution")

y_sol = ((np.exp(2)/(np.exp(4) - 1)) * (np.exp(2*t_p) - np.exp(-2*t_p))) + t_p
plt.plot(t_p, y_sol, 'b', label="Analytical solution")
plt.legend()
plt.xlabel("$x$")
plt.ylabel("$y(x)$")
plt.title("Solution of BVP")

# Defining relative error in Percentage, while taking care of division by zero
err = np.zeros(len(y_sol))
for i in range(len(y_sol)):
    if y_sol[i] == 0 and y_num[i] == 0:
        err[i] = 0
    elif y_sol[i] == 0 and y_num[i] != 0:
        err[i] = np.inf
    else:
        err[i] = 100 * (y_num[i] - y_sol[i]) / y_sol[i]

# Tabulating by writing to csv file
if os.path.exists('P_8.csv'):
    os.remove('P_8.csv')       # Removing previously existing table

with open('P_8.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["x", "y Numerical", "y Analytical", "y Error (%)"])
    writer.writerows(np.transpose(np.array([t_p, y_num, y_sol, err])))

plt.show()
