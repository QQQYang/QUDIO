#encoding=utf-8
"""
Implementation of classification circuit
--------------------------------------
Author: Yang Qian
Email: qianyang9@jd.com
"""

from pennylane import wires
import torch
import pennylane as qml
from pennylane.templates import AmplitudeEmbedding
from .embedding import CNOT_layer

import math
from pennylane import numpy as np

# import qiskit.providers.aer.noise as noise
# noise_model = noise.NoiseModel()

# # U1: RZ
# error_rate = 0.1
# error1 = noise.depolarizing_error(error_rate, num_qubits=1)
# error2 = noise.depolarizing_error(error_rate, num_qubits=2)
# noise_model.add_all_qubit_quantum_error(error1, ['u1', 'u2', 'u3'])
# noise_model.add_all_qubit_quantum_error(error2, ['cx'])

def classifier(param, feat=None, p=0):
    '''
    Implementation of classification circuit.
    -----------------------------------------
    :param param: learnable parameters of classifier, [n_layers, n_qubits, 3]
    :param feat: classical features, [n_layers, n_qubits]
    Return expectation value of Pauli-Z on qubit 1
    '''
    # AmplitudeEmbedding(features=feat, wires=range(6), normalize=True)
    # n_layers, n_qubits = param.size()[0], param.size()[1]
    n_layers, n_qubits = param.shape[0], param.shape[1]
    for i in range(n_layers):
        for j in range(n_qubits):
            qml.Rot(param[i, j, 0], param[i, j, 1], param[i, j, 2], wires=j)
            qml.DepolarizingChannel(p, wires=j)
        CNOT_layer(n_qubits, p)
    return qml.expval(qml.PauliZ(0))

# class QNN(torch.nn.Module):
#     def __init__(self, n_qubits, n_layers, p=0, param_shift=False):
#         super().__init__()

#         dev_noise = qml.device('qiskit.aer', wires=n_qubits, noise_model=noise_model, shots=10)
#         # dev_noise = qml.device('default.qubit', wires=n_qubits, shots=10)
#         self.cir_noise = qml.QNode(classifier, dev_noise, interface='torch')

#         self.n_layers = n_layers
#         self.param = torch.nn.Parameter(torch.FloatTensor(n_layers, n_qubits, 3).uniform_(0, 2*math.pi))
#         self.register_parameter('param', self.param)

#     @property
#     def device(self):
#         return self.param.device

#     def forward(self, feat):
#         return self.cir_noise(self.param, feat=torch.abs(feat))

class QNN:
    def __init__(self, n_qubits, n_layers, p=0, param_shift=False):
        # dev_noise = qml.device('qiskit.aer', wires=n_qubits, noise_model=noise_model, shots=10)
        dev_noise = qml.device('default.qubit', wires=n_qubits, shots=10)
        self.cir_noise = qml.QNode(classifier, dev_noise)

        self.n_layers = n_layers
        self.param = np.random.uniform(0, 2*math.pi, n_layers*n_qubits*3)
        self.param = np.reshape(self.param, (n_layers, n_qubits, 3))

    def __call__(self, feat, param):
        return self.cir_noise(param, feat=feat)