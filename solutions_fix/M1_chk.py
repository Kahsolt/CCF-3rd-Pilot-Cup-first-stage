#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/05/13

# 验证判题结果
# 我的选择: C, D (5分，判为少选)

import numpy as np
from pyvqnet.qnn.vqc import Hadamard, RZ, RX, PhaseShift

# 选项A: 直接计算可知不等
H_mat = Hadamard().matrix
phi = np.pi / 4
Rxzx_mat = PhaseShift(phi).matrix.numpy() @ RZ(phi).matrix.numpy() @ RX(phi).matrix.numpy() @ RZ(phi).matrix.numpy()

print('H_mat:')
print(H_mat)
print('Rxzx_mat:')
print(Rxzx_mat)
print('=' * 42)


# 选项B: 取特殊值可知不等
# NOTE: 可能vqc包里的 RZ 定义和文档里的不一样
theta = np.pi / 8
HRzH_mat = Hadamard().matrix.numpy() @ RZ(theta).matrix.numpy() @ Hadamard().matrix.numpy()
Rx_mat = RX(theta).matrix

print('HRzH_mat:')
print(HRzH_mat)
print('Rx_mat:')
print(Rx_mat)
print('=' * 42)

# Chap 3.5 Hadamard gates
# https://threeplusone.com/pubs/on_gates.pdf
