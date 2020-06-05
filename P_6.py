"""
Author: Pruthvi Suryadevara
Email:  pruthvi.suryadevara@tifr.res.in
Solving initial value problem using solve_ivp
solve_ivp have defeult relative and absolute tolerance of 1e-3, 1e-6
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as inte


def f1(x, y):  # Defining the pair of Differential equations
    dy1 = 32*y[0] + 66*y[1] + (2/3)*x + (2/3)
    dy2 = -66*y[0] - 133*y[1] - (1/3)*x - (1/3)
    return(np.array([dy1, dy2]))


# Solving using solve_ivp (using backward integration) and plotting
y = inte.solve_ivp(f1, [0, 0.5], [1/3, 1/3], dense_output=True, method='BDF')
t = np.linspace(0, 0.5, 100)  # creating t mesh for plotting
plt.plot(t, y.sol(t).T, label="Numerical Solution")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title(" Solution of IVP")
plt.legend(["$y1$", "$y2$"])

if y.success:   # Testing for sucess of solution
    print("Solution to IVP is within Tolerance value, so solution is correct")
else:
    print("Solution to IVP is not with Tolerance, Not correct")

plt.show()
