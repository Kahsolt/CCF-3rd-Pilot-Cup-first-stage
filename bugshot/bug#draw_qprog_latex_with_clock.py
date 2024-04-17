#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/04/23

from pyqpanda import *

# BUG: draw_qprog_latex_with_clock does not work

# ↓↓↓ Example ↓↓↓

qvm = CPUQVM()
qvm.init_qvm()
qv = qvm.qAlloc_many(2)
cv = qvm.cAlloc_many(1)

prog = QProg() \
  << H(qv) \
  << CNOT(qv[0], qv[1]) \
  << Measure(qv[0], cv[0])

# ↓ ok
print(draw_qprog_latex(prog))
# ↓ error
print(draw_qprog_latex_with_clock(prog))
print(draw_qprog_latex_with_clock(
  prog, 
  config_data='QPandaConfig.json', 
  auto_wrap_len=100, 
  output_file='QCircuit.tex', 
  with_logo=False,
))
