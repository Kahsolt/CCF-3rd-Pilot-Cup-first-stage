#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/04/16

# BUG: count_prog_info 计数 single_gate_num 时遗漏了 P 门


# ↓↓↓ Example ↓↓↓

import numpy as np
from pyqpanda import *

S_P0_GATE = [I, H, X, Y, Z, S, T, X1, Y1, Z1]
S_P1_GATE = [RX, RY, RZ, P, U1]
S_P2_GATE = [U2]
S_P3_GATE = [U3]
S_P4_GATE = [U4]
S_GATE = S_P0_GATE + S_P1_GATE + S_P2_GATE + S_P3_GATE + S_P4_GATE


vqm = CPUQVM()
vqm.init_qvm()
q = vqm.qAlloc()

cnt = 0
qcir = QCircuit()
for gate in S_GATE:
  cnt += 1
  if gate in S_P0_GATE:
    qcir << gate(q)
  if gate in S_P1_GATE:
    qcir << gate(q, *np.random.uniform(size=1).tolist())
  if gate in S_P2_GATE:
    qcir << gate(q, *np.random.uniform(size=2).tolist())
  if gate in S_P3_GATE:
    qcir << gate(q, *np.random.uniform(size=3).tolist())
  if gate in S_P4_GATE:
    qcir << gate(q, *np.random.uniform(size=4).tolist())

  qprog = QProg() << qcir
  info = count_prog_info(qprog)
  print(f'>> cur gate: {gate.__name__}, cnt: {cnt}, single_gate_num: {info.single_gate_num}')
