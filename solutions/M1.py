#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/04/15

from tiny_q import *

def testA():
  # https://quantumcomputing.stackexchange.com/questions/5261/hadamard-gate-as-a-product-of-r-x-r-z-and-a-phase
  phi = pi / 4
  return np.allclose((Ph(phi) * RZ(phi) * RX(phi) * RZ(phi)).v, H.v)

def testB():
  for phi in np.linspace(-2*pi, 2*pi, 360):
    if np.allclose(RX(phi).v, (H * RZ(phi) * H).v):
      return False
  return True

def testC():
  # CNOT is entangled, if there exists CNOT = A @ B, then (A@B)|xy> will
  # decouple to A|x> @ B|y>, against the entanglement
  return True

def testD():
  return np.allclose((Y * Z * Y.dagger).v, -Z.v)

choices = {
  'A': testA(),
  'B': testB(),
  'C': testC(),
  'D': testD(),
}

answers = []
for choice, ok in choices.items():
  if ok:
    answers.append(choice)

print()
print('Answer:', ''.join(answers))
