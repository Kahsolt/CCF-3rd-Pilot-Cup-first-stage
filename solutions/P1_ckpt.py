#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/04/17

# 查看训练完的参数，与手工设计的参数作对比，结论:
#  1. 在给定线路结构的情况下，解似乎是是唯一的
#  2. 参数越多收敛越慢，见可学习的 init_state 旋转角，可能是 lr 退火的问题

from P1 import *

method_cases = {
  'diogo': [
    'WState_diogo_VQCx',
    'WState_diogo_VQC1',
  ],
  'qiskit': [
    'WState_qiskit_VQC0',
    'WState_qiskit_VQC1',
    'WState_qiskit_VQC2',
    'WState_qiskit_VQC3',
  ]
}

def get_truth(method:str, n_qubits:int):
  if 'qiskit' in method:
    return [np.arccos(np.sqrt(1 / (i + 1))) for i in range(1, n_qubits)]
  if 'diogo' in method:
    return [np.arcsin(np.sqrt(1 / (n_qubits - i))) for i in range(n_qubits-1)]

for n_qubits in n_qubits_cases:
  print('=' * 32 + f' [Q={n_qubits}] ' + '=' * 32)
  for case, methods in method_cases.items():
    truth = get_truth(case, n_qubits)
    print(f'[truth] {round_list(truth)}')
    for method in methods:
      try:
        ckpt = load_parameters(CKPT_PATH / f'{method}_Q={n_qubits}.pth')
        if 'thetas' in ckpt:    # by theta
          params = []
          if 'theta0' in ckpt:
            params.append(ckpt['theta0'].item())
          params.extend(ckpt['thetas'].numpy().tolist())
        else:                   # by gate
          params = [p.item() for p in ckpt.values()]
        print(f'[{method}] {round_list(params)}')
      except: pass
    print()
