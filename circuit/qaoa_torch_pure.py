import torch
import torch.distributions.bernoulli as bernoulli
from torch.autograd import Function
import math
import numpy as np

from .basis_gate import RY, RX, RZ, X, Y, Z, kronecker, CNOT, Rot, I, H

def U_B(beta, n_wires=4):
    x = RX(beta)
    for _ in range(n_wires-1):
        x = kronecker(x, RX(beta))
    return x

def U_C(gamma, graph, n_wires=4):
    x = torch.eye(2**n_wires, dtype=torch.cfloat)
    for edge in graph:
        layer1 = torch.eye(2**edge[0], dtype=torch.cfloat) if edge[0] > 0 else 1
        layer1 = kronecker(layer1, CNOT(edge[0], edge[1]))
        if edge[1]+1 < n_wires:
            layer1 = kronecker(layer1, torch.eye(2**(n_wires - edge[1] - 1), dtype=torch.cfloat))
        
        layer2 = torch.eye(2**edge[1], dtype=torch.cfloat) if edge[1] > 0 else 1
        layer2 = kronecker(layer2, RZ(gamma))
        if edge[1]+1 < n_wires:
            layer2 = kronecker(layer2, torch.eye(2**(n_wires - edge[1] - 1), dtype=torch.cfloat))
        
        layer3 = torch.eye(2**edge[0], dtype=torch.cfloat) if edge[0] > 0 else 1
        layer3 = kronecker(layer3, CNOT(edge[0], edge[1]))
        if edge[1]+1 < n_wires:
            layer3 = kronecker(layer3, torch.eye(2**(n_wires - edge[1] - 1), dtype=torch.cfloat))

        x = layer3 @ layer2 @ layer1 @ x
    return x

def circuit(param, graph, n_qubits, p=0, M=100):
    state = torch.zeros((2**n_qubits, 1), dtype=torch.cfloat)
    gate = 1
    for _ in range(n_qubits):
        gate = kronecker(gate, H)
    state = gate @ state

    n_layers = param.size()[0]
    for i in range(n_layers):
        state = U_C(param[i, 0], graph, n_qubits) @ state
        state = U_B(param[i, 1], n_qubits) @ state

    # measurement
    prob = state[:, 0].abs()**2
    for edge in graph:
        prob_nodes = []
        for k in edge:
            prob_node = 0
            step = 2**(n_qubits-k)
            for i in range(2**k):
                start = i*step
                for j in range(2**(n_qubits-k-1)):
                    prob_node += prob[step+j]
            (1-p)**20*prob+(1-(1-p)**20)/2
            prob_nodes.append(prob_node)
