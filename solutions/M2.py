#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/04/15

from tiny_q import *
from scipy.linalg import expm

print('H:', H)
print('S:', S)
print('S.dagger:', S.dagger)

Y = np.asarray([
  [0, -i],
  [i,  0],
])
tgt = np.exp(i*pi/4) * expm(-i*Y*pi/4)
print('tgt:', tgt)

print('=' * 72)

choices = {
  'A': S * H * S * H * S,
  'B': S * H * S.dagger * H * S,
  'C': S * H * S * H * S.dagger,
  'D': H * S * H * S * H,
}

answers = []
for choice, gate in choices.items():
  print(gate.v)
  if not np.allclose(gate.v, tgt):    # 选择错误项
    answers.append(choice)

print()
print('Answer:', ''.join(answers))
