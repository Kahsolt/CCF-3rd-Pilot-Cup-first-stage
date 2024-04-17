#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/04/16

from pyvqnet.qnn.vqc import *

# The module 'pyvqnet.qnn.vqc' provides handy quantum gates API 
# in two forms paralled: objective & functionalm, for examples:

# objectives signatures
#   .__init__(self, has_params:bool=False, trainable:bool=False, init_params:Incomplete=None, wires:Incomplete=None, dtype=..., use_dagger:bool=False)
#   .forward(self, params:Incomplete=None, q_machine:Incomplete=None, wires:Incomplete=None) -> None
PauliX, RX, CNOT
# functionals signatures
#   (q_machine, wires, params:Incomplete=None, use_dagger:bool=False) -> None
paulix, rx, cnot

# However, where a gate 'has_params' is due to its nature, should not
# dependends on the input arguments
# The API design is heavy, and throw tons of errors

# ↓↓↓ Example ↓↓↓

PauliX(has_params=True, trainable=True)      # <- AttributeError: 'PauliX' object has no attribute 'build_params'
PauliX(has_params=False, trainable=True)     # <- AttributeError: 'PauliX' object has no attribute 'build_params'

vqm = QMachine(1)
paulix(vqm, wires=[0], params=QTensor([0.1]))   # <- ValueError: got parameters for non-parametric gate paulix.
