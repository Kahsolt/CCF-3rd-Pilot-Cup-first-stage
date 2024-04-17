#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/04/16

# BUG: circuit transcription not fully support P & CP gate


# ↓↓↓ Example ↓↓↓

from pyqpanda import *

nq = 4
vqm = CPUQVM()
vqm.init_qvm()
qv = vqm.qAlloc_many(nq)

qcir = QCircuit() \
  << P(qv[1], 0.1) \
  << CP(qv[3], qv[2], 1.2)

qprog = QProg() << qcir
print(qprog)

# QProg <-> OriginIR
try:
  ir = convert_qprog_to_originir(qprog, vqm)
  convert_originir_str_to_qprog(ir, vqm)
except: print('>> OriginIR failed')

# QProg <-> QSAM
try:
  qasm = convert_qprog_to_qasm(qprog, vqm)
  convert_qasm_string_to_qprog(qasm, vqm)
except: print('>> QSAM failed')

# QProg <-> Binary
try:
  bin = convert_qprog_to_binary(qprog, vqm)
  convert_binary_data_to_qprog(vqm, bin)
except: print('>> Binary failed')
