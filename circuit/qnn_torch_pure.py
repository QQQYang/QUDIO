import torch
import torch.distributions.bernoulli as bernoulli
from torch.autograd import Function
import math

from .basis_gate import RY, Z, kronecker, CNOT, Rot, I, State00

def CNOT_layer(n_qubits=2):
    gate1 = CNOT(wires=[0, 1])
    for i in range(2, n_qubits, 2):
        if i+1 < n_qubits:
            gate1 = kronecker(gate1, CNOT(wires=[i, i+1]))
        else:
            gate1 = kronecker(gate1, I)
    gate2 = CNOT(wires=[1, 2])
    gate2 = kronecker(I, gate2)
    for i in range(3, n_qubits, 2):
        if i+1 < n_qubits:
            gate2 = kronecker(gate2, CNOT(wires=[i, i+1]))
        else:
            gate2 = kronecker(gate2, I)
    return torch.mm(gate2, gate1)

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

def amplitude_embedding(data):
    return data.unsqueeze(1)

def classifier(param, feat, p=0, M=100):
    #state = embedding(feat)
    state = amplitude_embedding(feat)
    n_layers, n_qubits = param.size()[0], param.size()[1]
    for i in range(n_layers):
        x = 1
        for j in range(n_qubits):
            x = kronecker(x, Rot(param[i, j, 0], param[i, j, 1], param[i, j, 2]))
        state = CNOT_layer(n_qubits).to(feat.device) @ (x @ state)

    if p<=0 and M<=0:
        # noiseless expectations
        return sum(state[0:len(state):2,0].abs()**2)
        # exp = torch.conj(state[:2]).t() @ Z @ state[:2]
        # return exp[0, 0].real
    else:
        # noisy expectations
        prob = sum(state[0:len(state):2,0].abs()**2)
        #sample = torch.rand(n_shots)
        #exp = (2*sum(sample<=prob) - n_shots) / n_shots

        # sample = torch.rand(1000)
        # exp = torch.mean(torch.tanh(prob - sample))

        m = bernoulli.Bernoulli((1-p)**20*prob+(1-(1-p)**20)/(2**1))
        exp = torch.mean(m.sample((M,)))
        return exp

class CircuitFn(Function):
    @staticmethod
    def forward(ctx, input, param, shift, p, M):
        ctx.save_for_backward(input, param)
        ctx.shift = shift
        ctx.p = p
        ctx.M = M
        return classifier(param, input, p, M)

    @staticmethod
    def backward(ctx, grad_output):
        input, param = ctx.saved_tensors
        shape = param.data.shape
        grad_param = torch.zeros_like(param.data)
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    mask = torch.zeros_like(param.data)
                    mask[i, j, k] = ctx.shift
                    exp_right = classifier(param + mask, input, ctx.p, ctx.M)
                    exp_left = classifier(param - mask, input, ctx.p, ctx.M)
                    grad_param[i, j, k] = (exp_right - exp_left)*grad_output
        return None, grad_param, None, None, None

class QNN(torch.nn.Module):
    def __init__(self, n_qubits, n_layers, p=0, M=100, param_shift=False):
        super().__init__()

        self.p = p
        self.M = M
        self.param_shift = param_shift
        self.n_layers = n_layers
        self.param = torch.nn.Parameter(torch.FloatTensor(n_layers, n_qubits, 3).uniform_(0, 2*math.pi))
        self.register_parameter('param', self.param)

    @property
    def device(self):
        return self.param.device

    def forward(self, feat):
        feat = feat.to(self.device)
        if self.param_shift:
            return CircuitFn.apply(feat, self.param, math.pi / 2, self.p, self.M)
        else:
            return classifier(self.param, feat, self.p, self.M)
