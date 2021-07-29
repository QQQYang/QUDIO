#encoding=utf-8
"""
Implementation of qaunvolutional neural network
reference: 
https://pennylane.ai/qml/demos/tutorial_quanvolution.html
https://github.com/PlanQK/TrainableQuantumConvolution/blob/main/Demo%20Trainable%20Quantum%20Convolution.ipynb
--------------------------------------
Author: Yang Qian
Email: qianyang9@jd.com
"""
import pennylane as qml
from pennylane.templates import RandomLayers
from pennylane import numpy as np
import qiskit
import qiskit.providers.aer.noise as noise
import torch
import torch.nn.functional as F
from multiprocessing import Pool
import math

from .build import CIRCUIT_REGISTRY
from .basis_gate import RY, Z, kronecker, CRR

def qconv_kernel_torch(phi, params):
    '''
    Implementation of quantum convolution kernel
    ------------------------------------------
    :param phi: image pixels, [n_quibits]
    :param params: learnable convolutional weights, [n_layers * n_qubits]
    '''
    init_state = torch.zeros((2**len(phi), 1), dtype=torch.cfloat)
    init_state[0, 0] = 1
    layers = []
    x = kronecker(RY(phi[0]), RY(phi[1]))
    for j in range(2, len(phi)):
        x = kronecker(x, RY(phi[j]))
    layers.append(x)

    param_index = 0
    for i in range(int(np.log2(len(phi)))):
        x = 1
        for j in range(0, len(phi), 2**(i+1)):
            if j+2**i < len(phi):
                qubit = CRR(params[param_index+1], wires=[j+2**i, j], name='RX') @ kronecker(1, CRR(params[param_index], wires=[j+2**i, j], name='RZ'))
                x = kronecker(x, qubit)
                param_index += 2
        for k in range(j+2**i+1, len(phi)):
            x = kronecker(x, torch.eye(2))
        layers.append(x)
    for l in layers:
        init_state = l @ init_state

    if len(phi) % 2 == 1:
        x = kronecker(1, CRR(params[param_index], wires=[len(phi)-1, 0], name='RZ'))
        x = CRR(params[param_index+1], wires=[len(phi)-1, 0], name='RX') @ x
        init_state = x @ init_state
    exp = torch.conj(init_state[:2]).t() @ Z @ init_state[:2]
    return torch.nn.functional.relu(exp[0, 0].real)

def qconv_torch(inputs, params, n_in_channels=1, kernel_size=[2, 2], stride=None, padding=False):
    """
    Convolves the input image with many applications of the same quantum circuit.
    ------------------------------------------------------------------------
    :param inputs: input image, [C, H, W]
    :param params: learnable convolutional weights, [n_out_channels, n_layers * n_qubits]
    :param params_channel: learnable channel weights, [n_out_channels, n_in_channels]
    :param kernel_size: size of qconv kernel, [kernel_h, kernel_w]
    :param stride: step of qconv, [stride_h, stride_w]
    :param padding: whether padding, bool
    """
    if stride is None:
        stride = kernel_size
    if padding:
        padding_h = kernel_size[0] // 2
        padding_w = kernel_size[1] // 2
        inputs = F.pad(inputs, (padding_w, padding_w, padding_h, padding_h))
        
    in_h, in_w = inputs.shape[1], inputs.shape[2]
    out_h = (in_h - kernel_size[0]) // stride[0] + 1
    out_w = (in_w - kernel_size[1]) // stride[1] + 1
    out = torch.zeros((len(params), out_h, out_w))

    for j in range(0, in_h-kernel_size[0], stride[0]):
        for k in range(0, in_w-kernel_size[1], stride[1]):
            channel_results = []
            for c_in in range(inputs.shape[0]):
                img_patch = []
                for h in range(kernel_size[0]):
                    for w in range(kernel_size[1]):
                        img_patch.append(inputs[c_in, j+h, k+w])
                out_channel_results = []
                for i in range(len(params)):
                    out_channel_results.append(qconv_kernel_torch(torch.stack(img_patch), params[i]))
                channel_results.append(torch.stack(out_channel_results))
            channel_results = torch.stack(channel_results) # [n_in_channels, n_out_channels]
            for i in range(len(params)):
                if n_in_channels == 1:
                    out[i, j//stride[0], k//stride[1]] = channel_results[0, i]
                else:
                    out[i, j//stride[0], k//stride[1]] = qconv_kernel_torch(channel_results[:, i], params[i, -(n_in_channels*2-2):])
    return out

class QCNN_torch(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        n_quibit = sum(cfg.MODEL.QCNN.KERNEL_SIZE)
        self.DIM = cfg.MODEL.QCNN.DIM
        self.PADDING = cfg.MODEL.QCNN.PADDING
        self.KERNEL_SIZE = cfg.MODEL.QCNN.KERNEL_SIZE
        
        self.qconv_param = []
        for i in range(len(self.DIM)-1):
            # param_weight = np.random.uniform(0, 2*math.pi, self.DIM[i+1]*(n_quibit*2-2+self.DIM[i]*2-2))
            # param_weight = np.reshape(param_weight, (self.DIM[i+1], n_quibit*2-2+self.DIM[i]*2-2))
            param_weight = torch.nn.Parameter(torch.FloatTensor(self.DIM[i+1], n_quibit*2-2+self.DIM[i]*2-2).uniform_(0, 2*math.pi))
            self.register_parameter('layer'+str(i+1), param_weight)
            self.qconv_param.append(param_weight)

        # FC
        n_feat = n_feat = ((cfg.DATASET.IMAGE_SIZE[0]//(2**(len(self.DIM)-1)))*(cfg.DATASET.IMAGE_SIZE[1]//(2**(len(self.DIM)-1)))) *  self.DIM[-1]
        stdv = 1./n_feat
        # self.fc = np.random.uniform(-stdv, stdv, n_feat*10)
        # self.fc = np.reshape(self.fc, (n_feat, 10))
        # self.fc = torch.nn.Parameter(torch.FloatTensor(n_feat, 10).uniform_(-stdv, stdv))
        # self.fc = torch.nn.Sequential(
        #     torch.nn.Linear(n_feat, 64),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(64, 10)
        # )
        self.fc = torch.nn.Linear(n_feat, cfg.DATASET.N_CLASS)

    @property
    def device(self):
        return self.fc.device

    def forward(self, x):
        out = []
        for b in range(len(x)):
            bx = x[b]
            for i in range(len(self.DIM)-1):
                bx = qconv_torch(bx, self.qconv_param[i], n_in_channels=self.DIM[i], padding=self.PADDING, kernel_size=self.KERNEL_SIZE)
            # out.append(bx.flatten().unsqueeze(0) @ self.fc)
            out.append(bx.flatten())
        # return torch.cat(out, dim=0)
        return self.fc(torch.stack(out))

@CIRCUIT_REGISTRY.register()
def build_qcnn_torch_pure(cfg):
    return QCNN_torch(cfg)

if __name__=='__main__':
    print('test')