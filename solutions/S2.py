#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/04/15

from tiny_q import *
import numpy as np
from scipy.linalg import expm

def Rn(alpha:float, theta:float, n:list):
  A = n[0] * X.v + n[1] * Y.v + n[2] * Z.v 
  return np.exp(1j*alpha) * expm(-1j*theta*A/2)

pi = np.pi
sqrt2 = np.sqrt(2)

choices = {
  'A': [Rn(pi/2, pi,   [1/sqrt2, 0, 1/sqrt2]), Rn(pi/4, pi/2, [0, 0, 1])],
  'B': [Rn(pi/2, pi,   [1/sqrt2, 0, 1/sqrt2]), Rn(pi/4, pi,   [0, 0, 1])],
  'C': [Rn(pi/2, pi/2, [0, 1/sqrt2, 1/sqrt2]), Rn(pi/4, pi/2, [0, 0, 1])],
  'D': [Rn(pi/2, pi/2, [0, 1/sqrt2, 1/sqrt2]), Rn(pi/4, pi,   [0, 0, 1])],
}

answer = None
for choice, (H_hat, S_hat) in choices.items():
  if np.allclose(H_hat, H.v) and np.allclose(S_hat, S.v):
    answer = choice

print()
print('Answer:', answer)
print('=' * 72)
