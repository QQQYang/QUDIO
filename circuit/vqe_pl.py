import pennylane as qml
from pennylane import numpy as np
import torch
import math

PauliGate = {
    'X': qml.PauliX,
    'Y': qml.PauliY,
    'Z': qml.PauliZ,
    'I': qml.Identity,
}

def gen_h(hamiltonian):
    coefs = []
    obs = []
    for h in hamiltonian:
        coefs.append(hamiltonian[h])
        ob = PauliGate[h[0]](0)
        for i in range(1, len(h)):
            ob = ob @ PauliGate[h[i]](i)
        obs.append(ob)
    return qml.Hamiltonian(coefs, obs)

def CNOT_layer(n_qubits=2):
    for i in range(0, n_qubits, 2):
        if i+1 < n_qubits:
            qml.CNOT(wires=[i, i+1])
    for i in range(1, n_qubits, 2):
        if i+1 < n_qubits:
            qml.CNOT(wires=[i, i+1])

def circuit(param, A=None, p=0):
    n_layers, n_qubits = param.size()[0], param.size()[1]
    for i in range(n_layers):
        for j in range(n_qubits):
            qml.Rot(param[i, j, 0], param[i, j, 1], param[i, j, 2], wires=j)
        CNOT_layer(n_qubits)
    if p > 0:
        for j in range(n_qubits):
            qml.DepolarizingChannel(p, wires=j)
    return qml.expval(A)

def circuit_h2(param, H=None, p=0):
    qml.PauliX(0)
    qml.PauliX(1)
    for i in [0, 1, 2, 3]:
        qml.Rot(*param[i], wires=i)
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[2, 0])
    qml.CNOT(wires=[3, 1])
    if p > 0:
        for j in range(4):
            qml.DepolarizingChannel(p, wires=j)
    return qml.expval(H)

class VQE_HE(torch.nn.Module):
    def __init__(self, dev, n_qubits, hamiltonian, p=0, n_layers=2) -> None:
        super().__init__()

        self.param = torch.nn.Parameter(torch.FloatTensor(n_layers, n_qubits, 3).uniform_(0, 2*math.pi))
        self.register_parameter('param', self.param)

        self.cir = qml.qnode(dev, interface='torch')(circuit)
        self.hamiltonian = gen_h(hamiltonian)
        self.p = p

    def forward(self):
        return self.cir(self.param, A=self.hamiltonian, p=self.p)

class VQE_H2(torch.nn.Module):
    def __init__(self, dev, n_qubits, hamiltonian, p=0, n_layers=2) -> None:
        super().__init__()

        self.param = torch.nn.Parameter(torch.FloatTensor(n_qubits, 3).uniform_(0, 2*math.pi))
        self.register_parameter('param', self.param)

        self.cir = qml.qnode(dev, interface='torch')(circuit_h2)
        self.hamiltonian = gen_h(hamiltonian)
        self.p = p

    def forward(self):
        return self.cir(self.param, A=self.hamiltonian, p=self.p)