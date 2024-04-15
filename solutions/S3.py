#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/04/15

import numpy as np
from scipy.linalg import expm

Y = np.asarray([
  [0, -1j],
  [1j,  0],
])
k = 1 / np.sqrt(5)
expA = np.asarray([
  [ 1, 2],
  [-2, 1],
]) * k
print('tgt:', expA)

print('=' * 72)

choices = {
  'A': np.arcsin(k),
  'B': -np.arccos(k),
  'C': np.arccos(-k),
  'D': np.arccos(k),
}

answer = None
for choice, theta in choices.items():
  expA_hat = expm(-1j*theta*Y)
  print(expA_hat)
  if np.allclose(expA_hat, expA):
    answer = choice

print()
print('Answer:', answer)
print('=' * 72)


# 反解
expD, U = np.linalg.eig(expA)
invU = np.linalg.inv(U)
assert np.allclose(U @ np.diag(expD) @ invU, expA)
A = U @ np.diag(np.log(expD)) @ invU @ np.linalg.inv(-1j * Y)
print(A.round(4))
print('theta:', A[0][0].real)
