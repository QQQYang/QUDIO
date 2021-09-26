import torch
import torch.distributions.bernoulli as bernoulli
from torch.autograd import Function
import math
import numpy as np

from .basis_gate import RY, X, Y, Z, kronecker, CNOT, Rot, I

PauliGate = {
    'X': X,
    'Y': Y,
    'Z': Z,
    'I': I
}

PauliEigenVec = {
    'X': [
        torch.tensor([
            [math.sqrt(2)/2, math.sqrt(2)/2],
        ], dtype=torch.cfloat),
        torch.tensor([
            [math.sqrt(2)/2, -math.sqrt(2)/2],
        ], dtype=torch.cfloat)
    ],
    'Y': [
        torch.tensor([
            [-0.707j, math.sqrt(2)/2],
        ], dtype=torch.cfloat),
        torch.tensor([
            [math.sqrt(2)/2, 0.707j],
        ], dtype=torch.cfloat)
    ],
    'Z': [
        torch.tensor([
            [1, 0],
        ], dtype=torch.cfloat),
        torch.tensor([
            [0, 1],
        ], dtype=torch.cfloat)
    ],
    'I': [
        torch.tensor([
            [1, 0],
        ], dtype=torch.cfloat),
        torch.tensor([
            [0, 1],
        ], dtype=torch.cfloat)
    ]
}

PauliEigenVal = {
    'X': [1.0, -1.0],
    'Y': [1.0, -1.0],
    'Z': [1.0, -1.0],
    'I': [1.0, 1.0]
}

def CNOT_layer(n_qubits=2):
    gate1 = kronecker(kronecker(I, I), CNOT(wires=[2, 3]))
    gate2 = kronecker(CNOT(wires=[2, 0]), I)
    gate3 = kronecker(I, CNOT(wires=[3, 1]))
    return gate3 @ gate2 @ gate1

def embedding(data):
    n_layers, n_qubits = data.size()[0], data.size()[1]
    state = torch.zeros((2**n_qubits, 1), dtype=torch.cfloat)
    state[0, 0] = 1
    for i in range(n_layers):
        x = 1
        for j in range(n_qubits):
            x = kronecker(x, RY(data[i, j]))
        state = CNOT_layer(n_qubits).to(data.device) @ (x @ state)
    return state

def cal_eigenval_vector(hamiltonian):
    obs = {}
    for h in hamiltonian:
        coef = hamiltonian[h]
        eigen_vector = []
        eigen_value = []
        for j in range(2**len(h)):
            x = PauliEigenVec[h[0]][(j//(2**(len(h)-1)))%2]
            v = PauliEigenVal[h[0]][(j//(2**(len(h)-1)))%2]
            for i in range(1, len(h)):
                y = PauliEigenVec[h[i]][(j//(2**(len(h)-(i+1))))%2]
                x = torch.kron(x, y)
                v *= PauliEigenVal[h[i]][(j//(2**(len(h)-(i+1))))%2]
            eigen_vector.append(x)
            eigen_value.append(v)
        eigen_vector = torch.stack(eigen_vector)
        eigen_value = torch.tensor(eigen_value)
        obs[h] = [eigen_vector, eigen_value, coef]
    return obs

def sample_basis_states(n_shots, number_of_states, state_probability):
    basis_states = torch.arange(number_of_states)
    sample = torch.distributions.categorical.Categorical(state_probability)
    return sample.sample((n_shots,))
            

def circuit(param, hamiltonian, p=0, M=100):
    n_qubits = param.size()[0]
    state = torch.zeros((2**n_qubits, 1), dtype=torch.cfloat)
    state[-4, 0] = 1 # |1100>
    x = 1
    for i in range(n_qubits):
        x = kronecker(x, Rot(param[i, 0], param[i, 1], param[i, 2]))
    state = CNOT_layer(n_qubits) @ (x @ state)

    # measurement
    """
    https://math.stackexchange.com/questions/3309632/eigenvalues-of-tensor-product-of-matrices-question-about-general-properties
    1. Eigenvalues of tensor product of matrices equal the product of eigenvalues of each matrix
    2. Orthogonality: if eigenvectors of each matric are orthogonal, then the eigenvectors of tensor product of matrics are orthogonal too.
    """
    exp = 0
    for h in hamiltonian:
        coef = hamiltonian[h]
        x = PauliGate[h[0]]
        for i in range(1, len(h)):
            x = kronecker(x, PauliGate[h[i]])
        exp += coef*(torch.conj(state).t() @ x @ state)[0,0].real
    return exp
    # n_shots = 100

    # exp = 0
    # for h in hamiltonian:
    #     ## 1. calculate the probability under the computational basis of observables
    #     rotated_prob = hamiltonian[h][0] @ state    # [2**n, 1]
    #     rotated_prob = torch.abs(rotated_prob[:, 0])**2
    #     ## 2. sampling
    #     index = sample_basis_states(M, 2**n_qubits, rotated_prob[:,0])
    #     exp += hamiltonian[h][2] * torch.mean(hamiltonian[h][1][index])
    # return exp

class CircuitFn(Function):
    """
    ref: 
    1. https://qiskit.org/textbook/ch-machine-learning/machine-learning-qiskit-pytorch.html#training
    2. https://pytorch.org/docs/stable/notes/extending.html
    """
    @staticmethod
    def forward(ctx, param, hamiltonian, shift, p, M):
        ctx.save_for_backward(param)
        ctx.shift = shift
        ctx.p = p
        ctx.M = M
        ctx.hamiltonian = hamiltonian
        return circuit(param, hamiltonian, p, M)

    @staticmethod
    def backward(ctx, grad_output):
        param = ctx.saved_tensors[0]
        shape = param.data.shape
        grad_param = torch.zeros_like(param.data)
        for i in range(shape[0]):
            for j in range(shape[1]):
                mask = torch.zeros_like(param.data)
                mask[i, j] = ctx.shift
                exp_right = circuit(param + mask, ctx.hamiltonian, ctx.p, ctx.M)
                exp_left = circuit(param - mask, ctx.hamiltonian, ctx.p, ctx.M)
                grad_param[i, j] = (exp_right - exp_left)*grad_output
        return grad_param, None, None, None, None

class VQE(torch.nn.Module):
    def __init__(self, n_qubits, hamiltonian, p=0, M=100, param_shift=False):
        super().__init__()

        self.p = p
        self.M = M
        self.param_shift = param_shift
        if param_shift:
            self.hamiltonian = cal_eigenval_vector(hamiltonian)
        else:
            self.hamiltonian = hamiltonian
        self.param = torch.nn.Parameter(torch.FloatTensor(n_qubits, 3).uniform_(0, 2*math.pi))
        self.register_parameter('param', self.param)

    @property
    def device(self):
        return self.param.device

    def forward(self):
        if self.param_shift:
            return CircuitFn.apply(self.param, self.hamiltonian, math.pi / 2, self.p, self.M)
        else:
            return circuit(self.param, self.hamiltonian, self.p, self.M)