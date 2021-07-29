import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import GradientDescentOptimizer

import time
from multiprocessing import Pool
import copy
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

import dask

def circuit(params):
    n_wires = len(params)
    for j in range(1):
        for i in range(n_wires):
            qml.RX(params[i], wires=i)
        for i in range(n_wires):
            qml.CNOT(wires=[i, (i + 1) % n_wires])
    return qml.expval(qml.PauliZ(n_wires - 1))

# from dask.distributed import Client

n_wires = 4
dev = qml.device("default.qubit", wires=n_wires)
qnode1 = qml.QNode(circuit, dev)
qnode2 = qml.QNode(circuit, dev)
d_node1 = qml.grad(qnode1)
qnodes = qml.QNodeCollection([qnode1, qnode2])
params = np.random.random(n_wires)
feat = np.random.random(32, requires_grad=False)

optimizer = GradientDescentOptimizer(0.01)

def cal_grad(param):
    d_node, weight = param
    return d_node(weight)

# params = [0.54, 0.12]
def get_opt():
    """
    Get parameters passed by python script
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_cpu", type=int, default = 8)
    opt = parser.parse_args()
    return opt

def mse(param, circuit):
    predict = np.array([circuit(param) for i in range(1)])
    return np.mean(predict**2)

def cost(param):
    return optimizer.step(lambda v: mse(v, qnode1), param)

def cost_mp(inputs):
    opt, param, circuit = inputs
    return opt.step(lambda v: mse(v, circuit), param)

if __name__ == '__main__':
    opt = get_opt()
    library = 'multiprocess' # dask, multiprocess
    time_grad = []

    n_cpu = opt.n_cpu
    cpu = 2
    for _ in tqdm(range(2, 1000, 1), desc='threads'):
        result = []

        st = time.time()
        if library == 'dask':
            for i in range(cpu):
                result.append(dask.delayed(cost, pure=True, traverse=False)(params))
            with dask.config.set(num_workers=cpu):
                param_new = dask.compute(*result, scheduler='threads', traverse=False)
        elif library == 'multiprocess':
            with Pool(n_cpu) as pool:
                st = time.time()
                result = pool.map(cost_mp, [(optimizer, params, qnode1) for _ in range(n_cpu)])
        else:
            print('{} is not supported'.format(library))

        # st = time.time()
        # for i in range(cpu):
        #     result.append(dask.delayed(d_node1, pure=True, traverse=False)(params, feat=feat[i]))
        # st = time.time()
        # dask.compute(*result, scheduler='threads', traverse=False)
        # time_p = time.time() - st

        # st = time.time()
        # for _ in range(cpu):
        #    result.append(dask.delayed(d_node1)(params))
        # st = time.time()
        # dask.compute(*result, scheduler='processes')
        # time_p = time.time() - st

        # st = time.time()
        # for _ in range(cpu):
        #     cal_grad((d_node1, params))
        # time_s = time.time() - st
        # time_grad.append([time_p, time_s])

    # with Pool(n_cpu) as pool:
    #    st = time.time()
    #    result = pool.map(cal_grad, [(d_node1, params) for _ in range(n_cpu)])
    #time_grad.append(time.time() - st)

    #st = time.time()
    # for _ in range(n_cpu):
    #    cal_grad((d_node1, params))
    #time_grad.append(time.time() - st)
    #np.save('logs/random_circuit/time_grad'+str(n_cpu), time_grad)
