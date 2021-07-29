import pennylane as qml

def circuit(params):
    n_wires = len(params)
    for j in range(1):
        for i in range(n_wires):
            qml.RX(params[i], wires=i)
        for i in range(n_wires):
            qml.CNOT(wires=[i, (i + 1) % n_wires])
    return qml.expval(qml.PauliZ(n_wires - 1))