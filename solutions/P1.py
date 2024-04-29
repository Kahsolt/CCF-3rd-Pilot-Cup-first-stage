#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/04/15

from __future__ import annotations
import warnings ; warnings.filterwarnings(action='ignore', category=DeprecationWarning)

from pathlib import Path
from typing import Union, List, Dict
from inspect import signature

import numpy as np
from scipy.special import kl_div

DEBUG = False

BASE_PATH = Path(__file__).parent
CKPT_PATH = BASE_PATH / 'ckpt' ; CKPT_PATH.mkdir(exist_ok=True)

n_qubits_cases = [3, 4, 5, 6, 8, 10]
mean = lambda x: sum(x) / len(x) if len(x) else 0.0
round_list = lambda ls: [round(e, 7) for e in ls]

if 'vqi':
  import pyvqnet as vq
  # data
  from pyvqnet.tensor import QTensor, tensor
  from pyvqnet.dtype import kfloat32, kcomplex64
  # nn
  from pyvqnet.nn.parameter import Parameter
  from pyvqnet.nn.module import Module, ModuleList
  from pyvqnet.nn.loss import MeanSquaredError
  from pyvqnet.optim import SGD, Adagrad, Adadelta, RMSProp, Adam, Adamax
  from pyvqnet.utils import set_random_seed
  from pyvqnet.utils.initializer import ones, zeros, normal, uniform, quantum_uniform, he_normal, he_uniform, xavier_normal, xavier_uniform
  from pyvqnet.utils.metrics import MAE, MAPE, MSE, RMSE, SMAPE, R_Square
  from pyvqnet.utils.storage import save_parameters, load_parameters
  # qnn.vqc: new vqc interface
  ProbDict = Dict[str, float]
  StateDict = Dict[str, QTensor]
  Wires = Union[int, List[int]]
  from pyvqnet.qnn.vqc.qmachine import QMachine
  from pyvqnet.qnn.vqc.qmeasure import Probability  # (wires:Wires)
  # functional: directly evolves the QMachine's state
  from pyvqnet.qnn.vqc.qcircuit import (
    # basic gates: 
    #   .__call__(q_machine:QMachine, wires:Wires, params:ndarray=None, num_wires:int=None, use_dagger:bool=False)
    hadamard,
    paulix,
    pauliy,
    pauliz,
    x1,
    y1,
    z1,
    rx,
    ry,
    rz,
    rxx,
    ryy,
    rzx,
    rzz,
    s,
    t,
    p,    # 相位旋转
    u1,
    u2,
    u3,
    cnot,
    crx,
    cry,
    crz,
    cr,   # 受控相位旋转
    cz,
    swap,
    iswap,
    toffoli,
    # advanced gates
    VQC_CCZ,                    # (wires:Wires, q_machine:QMachine), 双受控Z
    VQC_Controlled_Hadamard,    # (wires:Wires, q_machine:QMachine)
    VQC_RotCircuit,             # (params:QTensor, q_machine:QMachine, wire:Wires), RZ-RY-RZ 旋转
    VQC_CRotCircuit,            # (para:QTensor, control_wire:Wires, rot_wire:Wires, q_machine:QMachine), 受控 RZ-RY-RZ 旋转
    VQC_CSWAPcircuit,           # (wires:Wires, q_machine:QMachine)
  )
  # objective module classes
  from pyvqnet.qnn.vqc.qcircuit import (
    # trainable variational gates modules (无参门请用 functional 形式)
    #   .__init__(has_params:bool=False, trainable:bool=False, init_params:ndarray=None, num_wires:int=None, wires:Wires=None, dtype:int=kcomplex64, use_dagger:bool=False)
    #   .forward(params:ndarray=None, q_machine:QMachine=None, wires:Wires=None)
    RX as RX_original,
    RY as RY_original,
    RZ as RZ_original,
    RXX as RXX_original,
    RYY as RYY_original,
    RZX as RZX_original,
    RZZ as RZZ_original,
    U1 as U1_original,
    U2 as U2_original,
    U3 as U3_original,
    CRX as CRX_original,
    CRY as CRY_original,
    CRZ as CRZ_original,
    CR as CR_original,
    iSWAP as iSWAP_original,
    # trainable variational circuit modules
    #   .forward(q_machine:QMachine)
    VQC_BasicEntanglerTemplate,     # (num_layers:int=1, num_qubits:int=1, rotation:str='RX', initial=None, dtype:int=None)
    VQC_StronglyEntanglingTemplate, # (num_layers:int=1, num_qubits:int=1, ranges=None, initial=None, dtype:int=None)
    VQC_HardwareEfficientAnsatz,    # (n_qubits:int, single_rot_gate_list:List['RX'|'RY'|'RZ'], entangle_gate:str='CNOT'|'CZ', entangle_rules:str='linear'|'all', depth=1, initial=None, dtype:int=None)
    ExpressiveEntanglingAnsatz,     # (type:int[1~19], num_wires:int, depth:int, name:str='')
  )
  if 'wrap trainable variational gate modules':
    p_zeros = lambda n=1, dtype=kcomplex64: QTensor([0.0]*n, requires_grad=True, dtype=dtype)

    RX  = lambda wires, **kwargs: RX_original (has_params=True, trainable=True, wires=wires, **kwargs)
    RY  = lambda wires, **kwargs: RY_original (has_params=True, trainable=True, wires=wires, **kwargs)
    RZ  = lambda wires, **kwargs: RZ_original (has_params=True, trainable=True, wires=wires, **kwargs)
    RXX = lambda wires, **kwargs: RXX_original(has_params=True, trainable=True, wires=wires, **kwargs)
    RYY = lambda wires, **kwargs: RYY_original(has_params=True, trainable=True, wires=wires, **kwargs)
    RZX = lambda wires, **kwargs: RZX_original(has_params=True, trainable=True, wires=wires, **kwargs)
    RZZ = lambda wires, **kwargs: RZZ_original(has_params=True, trainable=True, wires=wires, **kwargs)
    U1  = lambda wires, **kwargs: U1_original (has_params=True, trainable=True, wires=wires, **kwargs)
    U2  = lambda wires, **kwargs: U2_original (has_params=True, trainable=True, wires=wires, **kwargs)
    U3  = lambda wires, **kwargs: U3_original (has_params=True, trainable=True, wires=wires, **kwargs)
    CRX = lambda wires, **kwargs: CRX_original(has_params=True, trainable=True, wires=wires, **kwargs)
    CRY = lambda wires, **kwargs: CRY_original(has_params=True, trainable=True, wires=wires, **kwargs)
    CRZ = lambda wires, **kwargs: CRZ_original(has_params=True, trainable=True, wires=wires, **kwargs)
    CR  = lambda wires, **kwargs: CR_original (has_params=True, trainable=True, wires=wires, **kwargs)   # 受控相位旋转
    iSWAP = lambda wires, **kwargs: iSWAP_original(has_params=True, trainable=True, wires=wires, **kwargs)

    class CU3(Module):    # 受控任意角度旋转
      def __init__(self, wires:Wires):
        super().__init__()
        assert len(wires) == 2 and all([isinstance(wire, int) for wire in wires])
        self.wires = wires
        self.params = Parameter(shape=[3], initializer=zeros, dtype=kfloat32)
      def forward(self, vqm:QMachine):
        crx(vqm, wires=self.wires, params=self.params[0:1])
        cry(vqm, wires=self.wires, params=self.params[1:2])
        crz(vqm, wires=self.wires, params=self.params[2:3])


''' Model '''

class Model(Module):

  def __init__(self, n_qubits:int):
    super().__init__()

    self.n_qubits = n_qubits
    self.vqm = QMachine(n_qubits, kcomplex64)
    self.pmeas = Probability(wires=list(range(n_qubits)))

  def forward(self):
    raise NotImplementedError

  def get_prob(self) -> QTensor:
    self.vqm.reset_states(1)   # bs
    self.forward()
    probs = self.pmeas(self.vqm)
    return probs[0]
  
  def get_prob_wstate(self) -> ProbDict:
    wstate = {}
    probs = self.get_prob()
    for i in range(self.n_qubits):
      bits = bin(2**i)[2:].rjust(self.n_qubits, '0')
      wstate[bits] = probs[2**i].item()
    return wstate

  def load_ckpt(self, fp:Path):
    ckpt_dict: StateDict = load_parameters(fp)
    state_dict = self.state_dict()
    if 'check keys':
      missing_keys = state_dict.keys() - ckpt_dict.keys()
      if DEBUG and missing_keys: print('>> missing_keys:', missing_keys)
      redundant_keys = ckpt_dict.keys() - state_dict.keys()
      if DEBUG and redundant_keys: print('>> redundant_keys:', redundant_keys)
    for k in ckpt_dict: state_dict[k] = ckpt_dict[k]    # override with ckpt
    self.load_state_dict(state_dict)

  def save_ckpt(self, fp:Path):
    save_parameters(self.state_dict(), fp)

  def __str__(self) -> str:
    return self.__class__.__name__

''' Method: determined construction '''

class WState3_wikipedia(Model):

  '''
  https://en.wikipedia.org/wiki/W_state
  https://demonstrations.wolfram.com/ThreeQubitWStatesOnAQuantumComputer
  '''

  def __init__(self):
    super().__init__(3)

  def forward(self):
    vqm = self.vqm
    phi3 = 2*np.arccos(1/np.sqrt(3))
    ry(vqm, wires=0, params=QTensor([phi3]))
    VQC_Controlled_Hadamard(wires=[0, 1], q_machine=vqm)
    cnot(vqm, wires=[1, 2])
    cnot(vqm, wires=[0, 1])
    paulix(vqm, wires=0)

class WState4_toffoli(Model):

  ''' https://quantumcomputing.stackexchange.com/questions/4350/general-construction-of-w-n-state '''

  def __init__(self):
    super().__init__(4)

  def forward(self):
    vqm = self.vqm
    hadamard(vqm, wires=0)
    hadamard(vqm, wires=3)
    paulix(vqm, wires=0)
    paulix(vqm, wires=3)
    toffoli(vqm, wires=[0, 3, 1])
    paulix(vqm, wires=0)
    paulix(vqm, wires=3)
    toffoli(vqm, wires=[0, 3, 2])
    cnot(vqm, wires=[2, 0])
    cnot(vqm, wires=[2, 3])

class WState_qiskit(Model):

  '''
  https://arxiv.org/abs/1606.09290
  https://github.com/qiskit-community/qiskit-community-tutorials/blob/master/awards/teach_me_qiskit_2018/w_state/W%20State%201%20-%20Multi-Qubit%20Systems.ipynb
  '''

  def F_gate(self, i:int, j:int, n:int, k:int):
    vqm = self.vqm
    theta = np.arccos(np.sqrt(1/(n-k+1)))
    ry(vqm, wires=j, params=QTensor([-theta]))
    cz(vqm, wires=[j, i])     # move some portion of amplitude from j to i
    ry(vqm, wires=j, params=QTensor([theta]))

  def forward(self):
    vqm = self.vqm
    nq = self.n_qubits
    paulix(vqm, wires=nq-1)
    for i in range(nq - 1, 0, -1):    # F(n-1, n)
      self.F_gate(i, i - 1, nq, nq - i)
    for i in range(nq - 1, 0, -1):    # reversed CNOT(n, n-1)
      cnot(vqm, wires=[i - 1, i])

class WState_diogo(Model):

  '''
  https://arxiv.org/abs/1807.05572
  https://physics.stackexchange.com/questions/311743/quantum-circuit-for-a-3-qubit-w-rangle-state
  https://quantumcomputing.stackexchange.com/questions/15506/how-to-implement-a-circuit-preparing-the-three-qubit-w-state
  '''

  def block(self, i:int, j:int, p:float):
    vqm = self.vqm
    if '4 gate version':
      theta = np.arccos(np.sqrt(p))   # θ/2 in essay
      cnot(vqm, wires=[i, j])
      ry(vqm, wires=j, params=QTensor([-theta]))
      cnot(vqm, wires=[i, j])
      ry(vqm, wires=j, params=QTensor([theta]))
    else:
      theta = np.arcsin(np.sqrt(p))   # θ' in essay
      ry(vqm, wires=j, params=QTensor([theta]))
      cnot(vqm, wires=[i, j])
      ry(vqm, wires=j, params=QTensor([-theta]))
    cnot(vqm, wires=[j, i])

  def forward(self):
    vqm = self.vqm
    paulix(vqm, wires=0)
    for i in range(self.n_qubits-1):
      self.block(i, i + 1, 1 / (self.n_qubits - i))

''' Method: VQA '''

class AnsatzModel(Model):

  ''' pre-defined ansatz in VQNet '''

  def __init__(self, n_qubits:int, depth:int, ansatz:Module):
    super().__init__(n_qubits)
    self.depth = depth
    self.ansatz = ansatz

  def __str__(self) -> str:
    return super().__str__() + f'_Q={self.n_qubits}_D={self.depth}'

  def forward(self):
    self.ansatz(self.vqm)

class BET(AnsatzModel):

  def __init__(self, n_qubits:int, depth:int=2):
    ansatz = VQC_BasicEntanglerTemplate(depth, n_qubits, 'RY')
    super().__init__(n_qubits, depth, ansatz)

class SET(AnsatzModel):

  ''' https://arxiv.org/abs/1804.00633 '''

  def __init__(self, n_qubits:int, depth:int=2):
    ansatz = VQC_StronglyEntanglingTemplate(depth, n_qubits)
    super().__init__(n_qubits, depth, ansatz)

class HEA(AnsatzModel):

  ''' https://arxiv.org/abs/1704.05018 '''

  def __init__(self, n_qubits:int, depth:int=2):
    ansatz = VQC_HardwareEfficientAnsatz(n_qubits, ['RX', 'RY'], entangle_gate='CNOT', entangle_rules='linear', depth=depth)
    super().__init__(n_qubits, depth, ansatz)

class EEA(AnsatzModel):

  ''' https://arxiv.org/abs/1905.10876
  Expressibility:
    L=1: 9 < 1 < 2 < 16 < 3 < 18 < 10 < 12 < 15 < 17 < 4 < 11 < 7 < 8 < 19 < 5 < 13 < 14 < 6
    L=2: 9 < 1 < 10 < 15 < 16 < 3 < 18 < 7 < 17 < 4 < 8 < 2 < 12 < 11 < 19 < 5 < 13 < 14 < 6
  Entangling capability:
    L=*: 1 < 7 < 3 < 16 < 8 < 5 < 18 < 17 < 4 < 10 < 19 < 13 < 12 < 14 < 11 < 6 < 2 < 5 < 9
  Ansatz definitions:
    1  := RX-RZ
    2  := HEA(rot=[RX, RZ], entgl=CNOT, rule=linear)
    3  := HEA(rot=[RX, RZ], entgl=CRZ,  rule=linear)
    4  := HEA(rot=[RX, RZ], entgl=CRX,  rule=linear)
    18 := HEA(rot=[RX, RZ], entgl=CRZ,  rule=cyclic)
    19 := HEA(rot=[RX, RZ], entgl=CRX,  rule=cyclic)
    5  := HEA(rot=[RX, RZ], entgl=CRZ,  rule=all) + RX-RZ
    6  := HEA(rot=[RX, RZ], entgl=CRX,  rule=all) + RX-RZ
    13 := HEA(rot=[RY],     entgl=CRZ,  rule=bicyclic)
    14 := HEA(rot=[RY],     entgl=CRX,  rule=bicyclic)
    15 := HEA(rot=[RY],     entgl=CNOT, rule=bicyclic)
    11 := RY-RZ + CNOT{MERA} + RY-RZ{MERA} + CNOT{MERA}
    12 := RY-RZ + SWAP{MERA} + RY-RZ{MERA} + SWAP{MERA}
    7  := RX-RZ + CRZ{MERA} + RX-RZ + CRZ{MERA}
    8  := RX-RZ + CRX{MERA} + RX-RZ + CRX{MERA}
    16 := RX-RZ + CRZ{MERA} + CRX{MERA}
    17 := RX-RZ + CRZ{MERA} + CRX{MERA}
    9  := H + SWAP[linear] + RX
    10 := RY + SWAP[cyclic] + RY
  '''

  def __init__(self, n_qubits:int, depth:int=2):
    kind = 14
    ansatz = ExpressiveEntanglingAnsatz(kind, n_qubits, depth=depth)
    super().__init__(n_qubits, depth, ansatz)

class CCQC(Model):

  ''' https://arxiv.org/abs/1804.00633 '''

  def __init__(self, n_qubits:int):
    super().__init__(n_qubits)

    self.rot1   = ModuleList(U3(wires=i) for i in range(n_qubits))
    self.entgl1 = ModuleList(CU3(wires=[(i+1)%n_qubits, i]) for i in reversed(range(n_qubits)))
    self.rot2   = ModuleList(U3(wires=i) for i in range(n_qubits))
    entgl2, ctrl_bit = [], 0
    offset = 3 if n_qubits >= 4 else 1
    for _ in range(n_qubits):
      rot_bit = (ctrl_bit - offset + n_qubits) % n_qubits
      entgl2.append(CU3(wires=[ctrl_bit, rot_bit]))
      ctrl_bit = rot_bit
    self.entgl2 = ModuleList(entgl2)

  def __str__(self) -> str:
    return super().__str__() + f'_Q={self.n_qubits}'

  def forward(self):
    vqm = self.vqm
    for rot   in self.rot1:   rot(q_machine=vqm)
    for entgl in self.entgl1: entgl.forward(vqm)
    for rot   in self.rot2:   rot(q_machine=vqm)
    for entgl in self.entgl2: entgl.forward(vqm)

class AmplitudeDistributor(Model):

  ''' https://arxiv.org/abs/1912.01618 '''

  def __init__(self, n_qubits:int):
    super().__init__(n_qubits)

    self.iswap = ModuleList(iSWAP(wires=[i, i+1]) for i in range(n_qubits-1))

  def __str__(self) -> str:
    return super().__str__() + f'_Q={self.n_qubits}'

  def forward(self):
    vqm = self.vqm
    paulix(vqm, wires=0)
    for iswap in self.iswap: iswap(q_machine=vqm)


class WState_VQC(Model):

  ''' parametrized WState_* method '''

  def __str__(self) -> str:
    return super().__str__() + f'_Q={self.n_qubits}'

class WState_diogo_VQCx(WState_VQC):

  ''' parametrizing per-theta in WState_diogo
  The basic block in essay https://arxiv.org/abs/1807.05572 can be drawn as:
            CNOT     revCNOT
  |1>---------o----------x---
              |          |
  |0>--RY(θ)--x--RY(-θ)--o---
      Controlled-G(p)  rCNOT
  '''

  def __init__(self, n_qubits:int):
    super().__init__(n_qubits)

    self.thetas = Parameter(shape=[n_qubits-1], initializer=quantum_uniform, dtype=kfloat32)

  def block(self, i:int, j:int):
    vqm = self.vqm
    ry(vqm, wires=j, params=self.thetas[i:i+1])
    cnot(vqm, wires=[i, j])
    ry(vqm, wires=j, params=-self.thetas[i:i+1])
    cnot(vqm, wires=[j, i])

  def forward(self):
    vqm = self.vqm
    paulix(vqm, wires=0)
    for i in range(self.n_qubits - 1):
      self.block(i, i + 1)

class WState_diogo_VQCy(WState_VQC):

  ''' parametrizing per-theta in WState_diogo (symmetrical version)
  |1>----o----RY(θ')---
         |      |
  |0>---RY(θ)---o------
  '''

  def __init__(self, n_qubits:int):
    super().__init__(n_qubits)

    # 学出来 thetas2 全都接近 pi，说明就应该是 CNOT 门，WState_diogo_VQCx 完全合理
    self.thetas1 = Parameter(shape=[n_qubits-1], initializer=quantum_uniform, dtype=kfloat32)
    self.thetas2 = Parameter(shape=[n_qubits-1], initializer=quantum_uniform, dtype=kfloat32)

  def block(self, i:int, j:int):
    vqm = self.vqm
    cry(vqm, wires=[i, j], params=self.thetas1[i:i+1])
    cry(vqm, wires=[j, i], params=self.thetas2[i:i+1])

  def forward(self):
    vqm = self.vqm
    paulix(vqm, wires=0)
    for i in range(self.n_qubits - 1):
      self.block(i, i + 1)

class WState_diogo_VQC1(WState_VQC):

  ''' parametrizing per-gate in WState_diogo '''

  def __init__(self, n_qubits:int):
    super().__init__(n_qubits)

    self.cry = ModuleList(CRY(wires=[i, i+1]) for i in range(n_qubits - 1))

  def block(self, i:int, j:int):
    vqm = self.vqm
    self.cry[i](q_machine=vqm)    # Controlled-G(p) = CU3 = RY-CNOT-RY = CRY
    cnot(vqm, wires=[j, i])

  def forward(self):
    vqm = self.vqm
    paulix(vqm, wires=0)
    for i in range(self.n_qubits - 1):
      self.block(i, i + 1)

class WState_qiskit_VQC0(WState_VQC):

  ''' parametrizing per-theta in WState_qiskit '''

  def __init__(self, n_qubits:int):
    super().__init__(n_qubits)

    self.thetas = Parameter(shape=[n_qubits-1], initializer=quantum_uniform, dtype=kfloat32)

  def init_state(self):
    vqm = self.vqm
    nq = self.n_qubits
    paulix(vqm, wires=nq-1)

  def F_gate(self, i:int, j:int):
    vqm = self.vqm
    ry(vqm, wires=j, params=-self.thetas[j:j+1])
    cz(vqm, wires=[j, i])
    ry(vqm, wires=j, params=self.thetas[j:j+1])

  def forward(self):
    vqm = self.vqm
    nq = self.n_qubits
    self.init_state()
    for i in range(nq - 1, 0, -1):    # F(n-1, n)
      self.F_gate(i, i - 1)
    for i in range(nq - 1, 0, -1):    # reversed CNOT(n, n-1)
      cnot(vqm, wires=[i - 1, i])

class WState_qiskit_VQC1(WState_qiskit_VQC0):

  ''' parametrizing per-theta + init_state in WState_qiskit '''

  def __init__(self, n_qubits:int):
    super().__init__(n_qubits)

    self.theta0 = Parameter(shape=[1], initializer=quantum_uniform, dtype=kfloat32)

  def init_state(self):
    vqm = self.vqm
    nq = self.n_qubits
    ry(vqm, wires=nq-1, params=self.theta0)    # X -> RY

class WState_qiskit_VQC2(WState_VQC):

  ''' parametrizing per-rot-gate in WState_qiskit '''

  def __init__(self, n_qubits:int):
    super().__init__(n_qubits)

    self.ry_0 = RY(wires=n_qubits-1)
    self.ry_n = ModuleList(RY(wires=i) for i in range(n_qubits - 1))
    self.ry_p = ModuleList(RY(wires=i) for i in range(n_qubits - 1))

  def F_gate(self, i:int, j:int):
    vqm = self.vqm
    self.ry_n[j](q_machine=vqm)
    cz(vqm, wires=[j, i])
    self.ry_p[j](q_machine=vqm)

  def forward(self):
    vqm = self.vqm
    nq = self.n_qubits
    self.ry_0(q_machine=vqm)          # X -> RY
    for i in range(nq - 1, 0, -1):    # F(n-1, n)
      self.F_gate(i, i - 1)
    for i in range(nq - 1, 0, -1):    # reversed CNOT(n, n-1)
      cnot(vqm, wires=[i - 1, i])

class WState_qiskit_VQC3(WState_qiskit_VQC2):

  ''' parametrizing per-gate in WState_qiskit '''

  def __init__(self, n_qubits:int):
    super().__init__(n_qubits)

    self.crz = ModuleList(CRZ(wires=[i, i+1]) for i in range(n_qubits - 1))
    self.crx = ModuleList(CRX(wires=[i, i+1]) for i in range(n_qubits - 1))

  def F_gate(self, i:int, j:int):
    vqm = self.vqm
    self.ry_n[j](q_machine=vqm)
    self.crz [j](q_machine=vqm)
    self.ry_p[j](q_machine=vqm)

  def forward(self):
    vqm = self.vqm
    nq = self.n_qubits
    self.ry_0(q_machine=vqm)          # X -> RY
    for i in range(nq - 1, 0, -1):    # F(n-1, n)
      self.F_gate(i, i - 1)
    for i in range(nq - 1, 0, -1):    # reversed CNOT(n, n-1)
      self.crx[i - 1](q_machine=vqm)


''' Trainer '''

MODELS = {k: v for k, v in globals().items() if type(v) == type(Model) and issubclass(v, Model) and v not in [Model, AnsatzModel, WState_VQC]}

def get_truth(n_qubits:int) -> QTensor:
  pdist = tensor.zeros([2**n_qubits], dtype=kfloat32)
  for i in range(n_qubits):
    pdist[2**i] = 1 / n_qubits
  return pdist

def get_model(
    method:str, n_qubits:int, depth:int=2,
    n_iter:int=10000, loss:str='l2', optim:str='SGD', lr:float=0.2, sgd_momentum:float=0.8, sgd_nesterov:bool=False,
    seed:int=114514, train_if_no_exist:bool=True,
  ) -> Model:

  ''' env '''
  set_random_seed(seed)

  ''' model '''
  model_cls = MODELS[method]
  env = locals()    # avoid closure :(
  kwargs = {k: env[k] for k in signature(model_cls).parameters.keys()}
  model = model_cls(**kwargs)
  model.eval()
  param_cnt = sum([p.numel() for p in model.parameters() if p.requires_grad])
  if DEBUG:
    print(f'>> [{method}] depth: {depth}, n_qubits: {model.n_qubits}, n_params: {param_cnt}')
    assert param_cnt == sum([p.numel() for p in model.parameters()])
  if param_cnt == 0: return model

  ''' load '''
  ckpt_fp = CKPT_PATH / f'{model}.pth'
  if ckpt_fp.exists():
    model.load_ckpt(ckpt_fp)
    return model
  elif not train_if_no_exist: return

  ''' data '''
  pdist = get_truth(n_qubits)

  ''' optim & loss '''
  optim_cls = getattr(vq.optim, optim)
  if optim == 'SGD':
    optim = optim_cls(model.parameters(), lr=lr, momentum=sgd_momentum, nesterov=sgd_nesterov)
  else:
    optim = optim_cls(model.parameters(), lr=lr)
  if loss == 'l1':
    criterion = lambda x, y: tensor.abs(x - y).mean()
  elif loss == 'l2':
    criterion = MeanSquaredError()
  else: raise ValueError(f'unknown loss_fn: {loss}')

  ''' train '''
  ema_loss = None
  model.train()
  for steps in range(1, n_iter+1):
    optim.zero_grad()
    qdist = model.get_prob()
    loss = criterion(pdist, qdist)
    loss.backward()
    optim._step()

    kl = kl_div(pdist.numpy(), qdist.numpy()).sum()
    if DEBUG and steps % 200 == 0:
      ema_loss = (0.4 * ema_loss + 0.6 * loss.item()) if ema_loss is not None else loss.item()
      print(f'[Epoch {steps}/{n_iter}] kl_div: {kl:.5f}, l1_loss: {MAE(pdist, qdist):.5g}, ema(loss): {ema_loss:.5g}')
    if kl < 1e-9: break   # converge early
  model.eval()

  ''' save '''
  model.save_ckpt(ckpt_fp)

  return model


''' Solver '''

def question1(n_qubits:int) -> ProbDict:
  ''' Special case for n_qubits = 4 '''
  assert n_qubits == 4

  method = 'WState_diogo_VQC1'
  depth = -1

  model = get_model(method, n_qubits, depth)
  result = model.get_prob_wstate()
  return result

def question2(n_qubits:int) -> ProbDict:
  ''' Arbitary case for n_qubits '''

  method = 'WState_diogo_VQC1'
  depth = -1

  model = get_model(method, n_qubits, depth)
  result = model.get_prob_wstate()
  return result


''' Test Entry '''

def timer(fn):
  from time import time
  def wrapper(*args, **kwargs):
    start = time()
    r = fn(*args, **kwargs)
    end = time()
    print(f'[Timer]: {fn.__name__} took {end - start:.3f}s')
    return r
  return wrapper

@timer
def pretrain(n_qubits:int):
  global DEBUG
  DEBUG = True

  # TODO: tune these to find the optimal setting
  method = 'AmplitudeDistributor'
  depth = 1
  n_iter = 10000
  loss = 'l2'
  optim = 'SGD'
  lr = 0.2 * n_qubits**2   # 梯度随量子位增加而急剧减小，需要加♂大♂力♂度♂
  seed = 114514

  model = get_model(method, n_qubits, depth, n_iter, loss, optim, lr, seed=seed)
  result = model.get_prob_wstate()
  print(result)

@timer
def benchmark(n_qubits_cases:List[int], train_if_no_exist:bool=False):
  global DEBUG
  DEBUG = False

  TRUTHS = { n_qubits: get_truth(n_qubits) for n_qubits in n_qubits_cases }

  def run_depth_group(depth:int, methods:List[str]):
    print('=' * 32 + f' [D={depth}] ' + '=' * 32)
    for method in methods:
      error_list = []
      for n_qubits in n_qubits_cases:
        model = get_model(method, n_qubits, depth, train_if_no_exist=train_if_no_exist)
        if model is not None:
          qdist = model.get_prob()
          pdist = TRUTHS[n_qubits]
          err = MAE(pdist, qdist).item()
        else:
          err = float('inf')
        error_list.append(err)
      print(f'[{method}] {mean(error_list):.5f} | {round_list(error_list)}')
    print()

  FIX_DEPTH_METHODS = [
    'WState_diogo',
    'WState_diogo_VQCx',
    'WState_diogo_VQCy',
    'WState_diogo_VQC1',
    'WState_qiskit',
    'WState_qiskit_VQC0',
    'WState_qiskit_VQC1',
    'WState_qiskit_VQC2',
    'WState_qiskit_VQC3',
    'CCQC',
    'AmplitudeDistributor',
  ]
  run_depth_group(-1, FIX_DEPTH_METHODS)

  VAR_DEPTH_METHODS = [
    'BET',
    'SET',
    'HEA',
    'EEA',
  ]
  for depth in [1, 2, 3]:
    run_depth_group(depth, VAR_DEPTH_METHODS)


if __name__ == '__main__':
  for n_qubits in n_qubits_cases:
    pretrain(n_qubits)
  benchmark(n_qubits_cases, train_if_no_exist=False)
