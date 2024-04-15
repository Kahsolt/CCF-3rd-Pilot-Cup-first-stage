#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/04/15

from tiny_q import *

tgt = np.asarray([
  [1,  1,  1,  1],
  [1,  i, -1, -i],
  [1, -1,  1, -1],
  [1, -i, -1,  i],
]) / 2
print('tgt:', tgt)
assert np.allclose(QFT(2).v, tgt)   # 2-qubit QFT

print('=' * 72)

choices = {
  'A': SWAP * (I @ H) * Gate(np.kron(I.v, M0.v) + np.kron(S.v, M1.v)) * (H @ I),
  # 门构造不合法，一定错误
  #'B': SWAP * (I @ X) * Gate(np.kron(I.v, M1.v) + np.kron(S.v, M1.v)) * (H @ X),
  'C': SWAP * (X @ H) * Gate(np.kron(I.v, M0.v) + np.kron(S.v, M1.v)) * (H @ X),
  'D': SWAP * (I @ H) * Gate(np.kron(I.v, M1.v) + np.kron(S.v, M0.v)) * (H @ I),
}

answers = ['B']
for choice, gate in choices.items():
  print(gate.v)
  if not np.allclose(gate.v, tgt):    # 选择错误项
    answers.append(choice)

print()
print('Answer:', ''.join(answers))
