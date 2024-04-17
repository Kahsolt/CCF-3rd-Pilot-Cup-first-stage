#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/04/23

from time import sleep
from pyqpanda import *

qvm = CPUQVM()
qvm.init_qvm()
qv = qvm.qAlloc_many(1)
cv = qvm.cAlloc_many(1)

qc = QCircuit() << X(qv[0])
prog = QProg() << qc << measure_all(qv, cv)

qvm.async_run(prog)   # <- seg fault
while not qvm.is_async_finished():
  print('>> processed', qvm.get_processed_qgate_num())
  sleep(0.1)

res = qvm.get_async_result()
print(res)

print('Done')
