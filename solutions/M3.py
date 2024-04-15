#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/04/15

from tiny_q import *

k = 1 / np.sqrt(2)
p0 = State(k * (v0.v + i * v1.v))     # |i>
p1 = State(k * (v0.v - i * v1.v))     # |-i>

choices = {
  'A': [
    k * ((v0 @ v0).v + (v1 @ v1).v),
    k * ((h0 @ h0).v + (h1 @ h1).v),
  ],
  'B': [
    k * (v0.v + i * v1.v),
    k * (i * v1.v - v0.v),
  ],
  'C': [
    k * (h0.v + h1.v),
    v1.v,
  ],
  'D': [
    k * (p0.v + p1.v),
    k * (h1.v + h0.v),
  ],
}

answers = []
for choice, (vec1, vec2) in choices.items():
  print('vec1:', vec1)
  print('vec2:', vec2)
  if np.allclose(vec1, vec2):
    answers.append(choice)

print()
print('Answer:', ''.join(answers))
