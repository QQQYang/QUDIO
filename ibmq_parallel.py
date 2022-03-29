# https://pennylane.ai/blog/2021/05/how-to-parallelize-qnode-execution/
import asyncio
from matplotlib.pyplot import imshow
import qiskit
import pennylane as qml

from pennylane import numpy as np
import json
import time
import dask
from multiprocessing import Pool
import asyncio

# symbols = ["H", "H"]
# coordinates = np.array([0.0, 0.0, -0.6614, 0.0, 0.0, 0.6614])

# H, qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates)

# electrons = 2
# hf = qml.qchem.hf_state(electrons, qubits)

str2gate = {
    'I': qml.Identity,
    'X': qml.PauliX,
    'Y': qml.PauliY,
    'Z': qml.PauliZ
}

def split_H(hami_file):
    h1, h2, h3, h4 = [], [], [], []
    h1_coef, h2_coef, h3_coef, h4_coef = [], [], [], []
    with open(hami_file, 'r') as f:
        data = json.load(f)
    for i, (key, coef) in enumerate(data.items()):
        if i%4 == 0:
            h1.append(str2gate[key[0]](wires=0) @ str2gate[key[1]](wires=1) @ str2gate[key[2]](wires=2) @ str2gate[key[3]](wires=3))
            h1_coef.append(coef)
        elif i%4==1:
            h2.append(str2gate[key[0]](wires=0) @ str2gate[key[1]](wires=1) @ str2gate[key[2]](wires=2) @ str2gate[key[3]](wires=3))
            h2_coef.append(coef)
        elif i%4==2:
            h3.append(str2gate[key[0]](wires=0) @ str2gate[key[1]](wires=1) @ str2gate[key[2]](wires=2) @ str2gate[key[3]](wires=3))
            h3_coef.append(coef)
        else:
            h4.append(str2gate[key[0]](wires=0) @ str2gate[key[1]](wires=1) @ str2gate[key[2]](wires=2) @ str2gate[key[3]](wires=3))
            h4_coef.append(coef)
    return qml.Hamiltonian(h1_coef, h1), qml.Hamiltonian(h2_coef, h2), qml.Hamiltonian(h3_coef, h3), qml.Hamiltonian(h4_coef, h4)

def circuit(param, H=None):
    qml.BasisState(np.array([1, 1, 0, 0], requires_grad=False), wires=[0, 1, 2, 3])
    qml.DoubleExcitation(param, wires=[0, 1, 2, 3])
    return qml.expval(H)

def circuit_hea(param, H=None):
    qml.BasisState(np.array([1, 1, 0, 0], requires_grad=False), wires=[0, 1, 2, 3])
    for i in [0, 1, 2, 3]:
        qml.Rot(*param[i], wires=i)
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[2, 0])
    qml.CNOT(wires=[3, 1])
    return qml.expval(H)

h1, h2, h3, h4 = split_H('data/h20.3.json')

provider = qiskit.IBMQ.enable_account('cfc6a18e1dae52ff1b3a18c3ef85e4a38c073700aef1cf352966084eb2a02d3f8f941646a1b0f88e10eda795e8ed8512222a45d77be1049cfc41968f294087f0')

def device1(theta):
    # load account
    st = time.time()
    # provider = qiskit.IBMQ.enable_account('cfc6a18e1dae52ff1b3a18c3ef85e4a38c073700aef1cf352966084eb2a02d3f8f941646a1b0f88e10eda795e8ed8512222a45d77be1049cfc41968f294087f0')
    dev = qml.device('qiskit.ibmq', wires=4, backend='ibmq_belem', provider=provider)
    print('Begin to run on device1')
    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    theta, loss = opt.step_and_cost(lambda param: qml.qnode(dev)(circuit)(param, H=h1), theta)
    # loss = qml.qnode(dev)(circuit)(theta, H=h1)
    print('Device1: loss={}, time={}'.format(loss, time.time() - st))
    return theta, loss

    # qiskit.IBMQ.disable_account()

def device2(theta):
    st = time.time()
    # provider=qiskit.IBMQ.enable_account('318b09c2fee55824a18bf816b840992e5e39a2ba3e1b7d95deb462a5db6f6763cfe524363c6a1464cf1b731fca38df2314eaa57e2f025faee9ee036a5d165c3b')
    dev = qml.device('qiskit.ibmq', wires=4, backend='ibmq_bogota', provider=provider)
    print('Begin to run on device2')
    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    theta, loss = opt.step_and_cost(lambda param: qml.qnode(dev)(circuit)(param, H=h2), theta)
    # loss = qml.qnode(dev)(circuit)(theta, H=h2)
    print('Device2: loss={}, time={}'.format(loss, time.time() - st))
    return theta, loss

def device(param):
    theta, backend, h = param
    theta = np.array(theta, requires_grad=True)
    st = time.time()
    # provider=qiskit.IBMQ.enable_account('318b09c2fee55824a18bf816b840992e5e39a2ba3e1b7d95deb462a5db6f6763cfe524363c6a1464cf1b731fca38df2314eaa57e2f025faee9ee036a5d165c3b')
    dev = qml.device('qiskit.ibmq', wires=4, backend=backend, provider=provider)
    print('Begin to run on backend {}'.format(backend))
    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    theta, loss = opt.step_and_cost(lambda param: qml.qnode(dev)(circuit)(param, H=h), theta)
    # loss_pos = qml.qnode(dev)(circuit)(theta+np.pi/2, H=h)
    # loss_neg = qml.qnode(dev)(circuit)(theta-np.pi/2, H=h)
    # loss = qml.qnode(dev)(circuit)(theta, H=h)
    # grad = (loss_pos - loss_neg)/2
    # theta = theta - 0.4*grad
    print('Backend {}: loss={}, time={}'.format(backend, loss, time.time() - st))
    return (theta, loss)

async def multiple_tasks(theta):
    tasks = [asyncio.create_task(device1(theta)), asyncio.create_task(device2(theta))]
    params, losses = [], []
    for task in tasks:
        param, loss = await task
        params.append(param)
        losses.append(loss)
    return params, losses

if __name__ == '__main__':
    theta = np.array(0.0, requires_grad=True)
    for i in range(10):
        # params, losses = asyncio.run(multiple_tasks(theta))

        # task = [dask.delayed(device1)(theta), dask.delayed(device2)(theta)]
        # params, losses = dask.compute(*task, scheduler='processes')

        with Pool(4) as pool:
            res = pool.map(device, [(theta, 'ibmq_belem', h1), (theta, 'ibmq_bogota', h2), (theta, 'ibmq_quito', h3), (theta, 'ibmq_lima', h4)])
        theta = np.mean([data[0].numpy() for data in res])
        print('The {}-th step: theta={}, loss={}'.format(i, [data[0].numpy() for data in res], [data[1] for data in res]))
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(multiple_tasks())
    # loop.close()

    # task = [dask.delayed(device1)(), dask.delayed(device2)()]
    # dask.compute(*task, scheduler='processes')