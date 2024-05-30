#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/05/27

# BUG: PyQPanda 中 QOperator.get_matrix 获取线路所对应的矩阵不符合常规位序
# expected: 接口默认设置下硬符合流行教材定义习惯的比特位序
# actually: 默认相反，且 PyQPanda 未暴露接口以供用户调整

import numpy as np
from pyqpanda import *

nq = 2
N = 2**nq

qvm = CPUQVM()
qvm.init_qvm()
qv = qvm.qAlloc_many(nq)

# 符合 Control-Gate 的真值表形式：CU|1,x> = CU|1,U_f(x)>，高位为控制比特，低位为受控比特
qcir = QCircuit()
qcir << H(qv[0]).control(qv[1])
print(qcir)

# 用 QOperator.get_matrix 直接获取线路所对应的矩阵
# NOTE: 获取到的矩阵高低位是逆序的!!
mat_raw = QOperator(qcir).get_matrix()
mat = np.asarray(mat_raw).reshape(N, N)
print('[mat]')
print(mat.real.round(4))
print()

# 穷举制备每个计算基态，逐列投影出线路所对应的矩阵
# 符合 Control-Gate 的矩阵形式：左上角为 I 右下角为 U
mat_my = np.zeros([N, N], dtype=np.complex64)
for i in range(N):
  idx = i
  init_state = QCircuit()
  for j in range(nq):
    if idx & 1:
      init_state << X(qv[j])
    idx >>= 1
  qprog = QProg() << init_state << qcir
  qvm.directly_run(qprog)
  qs = np.asarray(qvm.get_qstate())
  mat_my[:, i] = qs
print('[mat_my]')
print(mat_my.real.round(4))
print()
