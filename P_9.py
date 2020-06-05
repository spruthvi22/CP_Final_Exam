"""
Author: Pruthvi Suryadevara
Email:  pruthvi.suryadevara@tifr.res.in
Finding Singular value decomposition
"""

import numpy as np


A1 = np.array([[2, 1], [1, 0], [0, 1]])
[U1, s1, V1] = np.linalg.svd(A1)   # Finding SVD using library function

S1 = np.zeros(np.shape(A1))   # Making S matrix form s values
for i in range(len(s1)):
    S1[i, i] = s1[i]

# Printing U, S, V, and checking A = U.S.V*
print("U1 = \n", U1)
print("S1 = \n", S1)
print("V1 = \n", np.transpose(V1))
print("A1 = U1.S1.V1* = \n", np.matmul(U1, np.matmul(S1, V1)))

# Solving for second matrix
A2 = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]])
[U2, s2, V2] = np.linalg.svd(A2)

S2 = np.zeros(np.shape(A2))
for i in range(len(s2)):
    S2[i, i] = s2[i]

print("U2 = \n", U2)
print("S2 = \n", S2)
print("V2 = \n", np.transpose(V2))
print("A2 = U2.S2.V2* = \n", np.matmul(U2, np.matmul(S2, V2)))
