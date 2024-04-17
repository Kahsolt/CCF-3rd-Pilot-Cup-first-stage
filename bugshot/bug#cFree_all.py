#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/04/16

import random
from pyqpanda import *

random.seed(42)

vqm = CPUQVM()
vqm.init_qvm()
vqm.set_configure(max_qubit=72, max_cbit=72)

free_cbit = 72
alloc_cbits = []
free_cbits_ids = []
for _ in range(1000):
  if random.random() < 0.5:     # alloc
    if free_cbit <= 0: continue
    cnt = random.randrange(1, free_cbit+1)
    cbits = vqm.cAlloc_many(cnt)
    alloc_cbits.extend(cbits)
    free_cbit -= cnt
  else:                         # free
    if len(alloc_cbits) < 0: continue
    cnt = min(1, int(len(alloc_cbits) * random.random()))
    cbits = random.sample(alloc_cbits, cnt)

    try:
      vqm.cFree_all(cbits)
    except Exception as e:
      for cbit in cbits:
        if id(cbit) in free_cbits_ids:
          print(f'>> Double free of python-id: {id(cbit)}')
      raise e

    for cbit in cbits:
      alloc_cbits.remove(cbit)
      free_cbits_ids.append(id(cbit))
    free_cbit += cnt

print('Done')
