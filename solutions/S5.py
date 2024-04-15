#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/04/15

from tiny_q import *

print('X:', X)
print('H:', H)
print('T:', T)

print('=' * 72)

C = np.exp(i*pi/4)
tgt = np.asarray([
  [1 + C, C - i],
  [1 - C, C + i],
]) / 2
print('tgt:', tgt)

print('=' * 72)

choices = {
  'A': T * H * T * H,
  'B': H * T * H * T * X,
  'C': X * H * T * H * T,
  'D': H * T * H * T,
}

answer = None
for choice, gate in choices.items():
  print(gate.v)
  if np.allclose(gate.v, tgt):
    answer = choice

print()
print('Answer:', answer)
print('=' * 72)
