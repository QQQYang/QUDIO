import pennylane as qml
import torch
from pennylane import numpy as np
from pennylane.optimize import GradientDescentOptimizer
import math

dev = qml.device('qiskit.aer', wires=6, shots=10)


def cost(param, feat, label, model):
    loss = []
    for m in range(len(feat)):
        predict = model(param, data=feat[m])
        loss.append((label[m] - predict)**2)
    return np.mean(np.array(loss))

def circuit3(weights, data=None):
    qml.templates.AmplitudeEmbedding(features=data, wires=range(6), normalize=True)
    n_layers, n_qubits = weights.shape[0], weights.shape[1]
    for i in range(n_layers):
        for j in range(n_qubits):
            qml.Rot(weights[i, j, 0], weights[i, j, 1], weights[i, j, 2], wires=j)
    return qml.expval(qml.PauliZ(0))

weights = np.random.random([4, 6, 3])
data = np.random.random([2, 64], requires_grad=False)
label = np.array([1, -1], requires_grad=False)
qnode = qml.QNode(circuit3, dev)

optimizer = GradientDescentOptimizer(1e-2)
weights = optimizer.step(lambda v: cost(v, data, label, qnode), weights)