import numpy as np
import pylab
import copy
from qiskit import BasicAer
from qiskit.aqua import aqua_globals, QuantumInstance
from qiskit.aqua.algorithms import ExactEigensolver, VQE
from qiskit.aqua.components.optimizers import COBYLA
# from qiskit.chemistry.aqua_extensions.components.initial_states import HartreeFock
# from qiskit.chemistry.aqua_extensions.components.variational_forms import UCCSD
from qiskit.chemistry.drivers import PySCFDriver
from qiskit.chemistry.core import Hamiltonian, QubitMappingType

molecule = 'H .0 .0 -{0}; H .0 .0 {0}'
algorithms = ['ExactEigensolver']
start = 0.3  # Start distance
by    = 0.2  # How much to increase distance by
steps = 10   # Number of steps to increase by
energies = np.empty([len(algorithms), steps+1])
hf_energies = np.empty(steps+1)
distances = np.empty(steps+1)
eval_counts = np.empty(steps+1)

print('Processing step __', end='')
for i in range(steps+1):
    print('\b\b{:2d}'.format(i), end='', flush=True)
    d = start + i*by/steps
for j in range(len(algorithms)):
    driver = PySCFDriver(molecule.format(d/2), basis='sto3g')
    qmolecule = driver.run()
    operator =  Hamiltonian(qubit_mapping=QubitMappingType.JORDAN_WIGNER,
                            two_qubit_reduction=False)
    qubit_op, aux_ops = operator.run(qmolecule)

    if algorithms[j] == 'ExactEigensolver':
        result = ExactEigensolver(qubit_op).run()
    
    # define the variables for our results and print them
    lines, result = operator.process_algorithm_result(result)
    energies[j][i] = result['energy']
    hf_energies[i] = result['hf_energy']
    if algorithms[j] == 'VQE':
        eval_counts[i] = result['algorithm_retvals']['eval_count']
        distances[i] = d
    print(' --- complete')
    print('Distances: ', distances) #interatmomic distance
    print('Energies:', energies) # ground state energy calculated by the exact eigensolver and the VQE
    print('Hartree-Fock energies:', hf_energies) # energies calculated by the Hartree Fock algorithm
    print('VQE num evaluations:', eval_counts)