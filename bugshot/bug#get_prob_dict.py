#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/04/19

# BUG: directly call get_prob_dict()/*_list()/*_tuple() will cause SegmentFault

# expected: throw Exceptions, or remove these API if actually deprecated
# actually: causing SegmentFault

# ↓↓↓ Example ↓↓↓

from pyqpanda import *

qvm = CPUQVM()
qvm.init_qvm()

q = qvm.qAlloc()
prog = QProg() << H(q)
#r = qvm.prob_run_dict(prog, [q])   # if you do not call this first
#print(r)
r = qvm.get_prob_dict([q])      # <- will get SegmentFault
print(r)

print('Ok')
