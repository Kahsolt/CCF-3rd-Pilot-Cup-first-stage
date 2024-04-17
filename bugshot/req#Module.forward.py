#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/04/16

# BUG:
# expected: Module.forward() should accept no positional args
# actually: Module.forward() accepts at lease one positional args 'x'
# The signature for Module.forward is:
#   @abstractmethod
#   def forward(self, x, *args, **kwargs): ...


# ↓↓↓ Example ↓↓↓

from pyvqnet.optim import SGD
from pyvqnet.nn.loss import MeanSquaredError
from pyvqnet.qnn.vqc import *

assert Module.forward 

# We wanna train a prue ansatz circuit that outputing desired prob distribution
class Model(Module):

  def __init__(self, n_qubits:int):
    super().__init__()

    self.bs = 1
    self.vqm = QMachine(n_qubits)
    self.ansatz = VQC_HardwareEfficientAnsatz(n_qubits, ['RX', 'RY'], depth=2)
    self.pmeas = Probability(wires=list(range(n_qubits)))

  # This is a pure ansatz module, we do not have encoders, do not need input QTensor data
  # ↓↓↓ Hence this arg x is dummy, only to satisfy the function signature
  def forward(self, x:QTensor=None) -> QTensor:
    self.vqm.reset_states(self.bs)
    self.ansatz(self.vqm)
    probs = self.pmeas(self.vqm)
    return probs[0]

n_qubits = 2
target = QTensor([1/3, 1/3, 1/3, 0])
model = Model(n_qubits)
criterion = MeanSquaredError()
optim = SGD(model.parameters(), lr=0.1)

for i in range(10000):
  optim.zero_grad()
  #output = model()         # <- TypeError: __call__() takes at least 2 positional arguments (1 given)
  output = model(None)      # You have to pass some nonsense only to make it work, looks stupid!!
  loss = criterion(target, output)
  loss.backward()
  optim.step()

  if i % 100 == 0:
    print('loss:', loss.item())
