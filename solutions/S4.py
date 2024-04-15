#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/04/15

from tiny_q import *

q = State(np.asarray([np.sqrt(2), 1, 0, -1]) / 2)
M1 = MeasureOp(np.asarray([
  [1, -i, 0, 1],
  [i,  1, 0, i],
  [0,  0, 0, 0],
  [1, -i, 0, 1],
]) / 3)

prob_hat = q > M1
print('prob_hat:', prob_hat)

choices = {
  'A': np.sqrt(2) / 6,
  'B': (2 + np.sqrt(2)) / 6,
  'C': (2 - np.sqrt(2)) / 6,
  'D': np.sqrt(2) / 3,
}

answer = None
for choice, prob in choices.items():
  print(f'{choice}: {prob}')
  if np.isclose(prob_hat, prob):
    answer = choice

print()
print('Answer:', answer)
