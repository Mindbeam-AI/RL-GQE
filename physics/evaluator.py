import numpy as np
import pennylane as qml

def build_operator_pool(n_qubits, t_values=None):

    if t_values is None:
        # t_values = [np.pi, np.pi/2, np.pi/3, np.pi/4, np.pi/8]
        t_values = [np.pi, np.pi/2, np.pi/4, np.pi/8, np.pi/16, np.pi/32]
        t_values += [-t for t in t_values]  # add negatives

    pool = []

    # Two-qubit interactions (Z_i Z_{i+1})
    for i in range(n_qubits - 1):
        for t in t_values:
            pool.append(qml.PauliRot(t, 'ZZ', wires=[i, i + 1]))
            pool.append(qml.PauliRot(t, 'XX', wires=[i, i+1]))
            pool.append(qml.PauliRot(t, 'YY', wires=[i, i+1]))

    # Single-qubit terms (X_i)
    for i in range(n_qubits):
        for t in t_values:
            pool.append(qml.PauliRot(t, 'X', wires=[i]))
            pool.append(qml.PauliRot(t, 'Y', wires=i))
            pool.append(qml.PauliRot(t, 'Z', wires=i))
            
    return pool

dev = qml.device("default.qubit", wires=4)

@qml.qnode(dev)
def energy_circuit(seq, ham, init_state, num_qubits):
    qml.BasisState(init_state, wires=range(num_qubits))

    for op in seq:
        qml.Snapshot(measurement=qml.expval(ham))
        qml.apply(op)
    return qml.expval(ham)

energy_circuit = qml.snapshots(energy_circuit)

def get_subsequence_energies(seq, hamiltonian, init_state, num_qubits):
    energies = []
    for pool in seq:
        es = energy_circuit(pool, hamiltonian, init_state, num_qubits)
        energies.append(
            [es[k].item() for k in list(range(1, len(pool))) + ["execution_results"]]
        )
    return np.array(energies, dtype=np.float32)
