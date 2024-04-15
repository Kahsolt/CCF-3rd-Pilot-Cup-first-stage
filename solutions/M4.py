#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/04/15

from tiny_q import *

rCNOT = Gate([
  [1, 0, 0, 0],
  [0, 0, 0, 1],
  [0, 0, 1, 0],
  [0, 1, 0, 0],
])

q = v0 @ v1 @ v1 @ v0
#GHZ_circ = (I @ I @ CNOT) * (I @ CNOT @ I) * (CNOT @ I @ I) * (H @ I @ I @ I)
GHZ_circ = (rCNOT @ I @ I) * (I @ rCNOT @ I) * (I @ I @ rCNOT) *(I @ I @ I @ H)

psi = GHZ_circ | q
print('psi.prob', psi.prob)

choices = {
  'A': '0110',
  'B': '1001',
  'C': '0010',
  'D': '1101',
}
inv_choices = {v: k for  k, v in choices.items()}

answers = []
for i, p in enumerate(psi.prob):
  if p > 0:
    bits = bin(i)[2:].rjust(4, '0')
    print(bits)
    answers.append(inv_choices[bits])

print()
print('Answer:', ''.join(answers))
