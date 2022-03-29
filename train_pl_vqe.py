import time
from pennylane import numpy as np
import pennylane as qml
#from pennylane import qchem
import os
import sys
import json

data = {  # keys: atomic separations (in Angstroms), values: corresponding files
    0.3: "./data/LiH_0.3.xyz",
    0.5: "./data/LiH_0.5.xyz",
    0.7: "./data/LiH_0.7.xyz",
    0.9: "./data/LiH_0.9.xyz",
    1.1: "./data/LiH_1.1.xyz",
    1.3: "./data/LiH_1.3.xyz",
    1.5: "./data/LiH_1.5.xyz",
    1.7: "./data/LiH_1.7.xyz",
    1.9: "./data/LiH_1.9.xyz",
    2.1: "./data/LiH_2.1.xyz",
}


hamiltonians = []
for separation, file in data.items():
    print(separation)
    symbols, coordinates = qml.qchem.read_structure(file)
    h, qubits = qml.qchem.molecular_hamiltonian(
        symbols=symbols,
        name='LiH'+str(separation),
        coordinates=coordinates,
        charge=0,
        mult=1,
        basis='sto-3g',
        mapping='jordan_wigner'
    )
    hamiltonians.append(h)

def circuit(param, wires ):
    qml.BasisState(np.array([1, 1, 0, 0], requires_grad=False), wires=[0, 1, 2, 3])
    for i in wires:
        qml.Rot(*param[i], wires=i)
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[2, 0])
    qml.CNOT(wires=[3, 1])

def CNOT_layer(n_qubits=2):
    for i in range(0, n_qubits, 2):
        if i+1 < n_qubits:
            qml.CNOT(wires=[i, i+1])
    for i in range(1, n_qubits, 2):
        if i+1 < n_qubits:
            qml.CNOT(wires=[i, i+1])

def hardware_efficient_ansatz(param, wires):
    n_layer = len(param)
    for i in range(n_layer):
        for j in wires:
            qml.Rot(*param[i, j], wires=j)
        CNOT_layer(len(wires))


qubits = 12
n_layer = 2
dev = qml.device('default.qubit', wires= qubits)
energies = [qml.ExpvalCost(hardware_efficient_ansatz, h, dev) for h in hamiltonians]
opt = qml.GradientDescentOptimizer(stepsize=0.4)


max_iterations = 300
conv_tol = 1e-06
dict_record = {}
def calculate_surface(parallel=False):
    s = []
    for i, e in enumerate(energies):
        print("Running for inter-atomic distance {} Å".format(list(data.keys())[i]))
        np.random.seed(0)
        params = np.random.normal(0, np.pi, (n_layer, qubits, 3))
        for n in range(max_iterations):
            params, prev_energy = opt.step_and_cost(e, params)
            energy = e(params)
            conv = np.abs(energy - prev_energy)

            if n % 20 == 0:
                print('Iteration = {:},  Energy = {:.8f} Ha'.format(n, energy))

            if conv <= conv_tol:
                break
        s.append(energy)
        dict_record['bond-distance'+str(list(data.keys())[i])]  = params
    return s

surface_seq = calculate_surface(parallel=True)
print(surface_seq)

np.save("energy_VQE_bench", surface_seq)
f = open('dict-bench.txt','w')
f.write(str(dict_record))
f.close()
