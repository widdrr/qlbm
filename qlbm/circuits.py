import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Gate
from qiskit.circuit.library import HGate
from qiskit.quantum_info import Statevector


def r_gate(num_qubits: int) -> Gate:
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits - 1, -1, -1):
        if i == 0:
            qc.x(i)
        else:
            controls = list(range(0, i))
            qc.mcx(controls, i)

    return qc.to_gate(label='R')


def l_gate(num_qubits: int) -> Gate:
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        if i == 0:
            qc.x(i)
        else:
            controls = list(range(0, i))
            qc.mcx(controls, i)

    return qc.to_gate(label='L')


def encode_links(link_qubits: int, num_links: int) -> QuantumCircuit:
    qc = QuantumCircuit(link_qubits)
    for i in range(link_qubits):
        qc.h(i)
    return qc


def encode(variable: NDArray[np.float64], link_qubits: int, extra_ancillas: int = 0) -> Statevector:
    state = variable.flatten(order='F')
    if not (len(state) & (len(state) - 1) == 0):
        raise ValueError("Length of flattened array must be a power of 2")

    state = Statevector(state / np.linalg.norm(state))
    ancilla = Statevector([1.0, 0.0])
    for _ in range(link_qubits):
        state = state.expand(ancilla)

    for _ in range(1 + extra_ancillas):
        state = state.expand(ancilla)

    return state


def macros(link_qubits: int, extra_ancillas: int = 0) -> QuantumCircuit:
    num_qubits = link_qubits + 1 + extra_ancillas
    qc = QuantumCircuit(num_qubits)
    ancilla = link_qubits

    for i in range(link_qubits):
        qc.swap(i, ancilla)
        qc.h(ancilla)

    # extra ancillas (e.g. BC LCU ancilla) are left untouched —
    # their |0⟩ state encodes LCU success and must not be mixed

    return qc


def propagation(site_qubits: list[int], link_qubits: int, links: list[list[int]]) -> QuantumCircuit:
    tot_site_qubits = np.sum(site_qubits)
    num_qubits = tot_site_qubits + link_qubits
    qc = QuantumCircuit(num_qubits)

    targets = [list(range(sum(site_qubits[:i]), sum(site_qubits[:i + 1]))) for i in range(len(site_qubits))]
    if len(links) == 1:
        for j, dir in enumerate(links[0]):
            while dir < 0:
                qc.append(l_gate(site_qubits[j]), targets[j])
                dir += 1
            while dir > 0:
                qc.append(r_gate(site_qubits[j]), targets[j])
                dir -= 1
        return qc

    controls = list(range(tot_site_qubits, num_qubits))
    for i, link in enumerate(links):
        control_state = np.binary_repr(i, link_qubits)
        for j, dir in enumerate(link):
            while dir < 0:
                qc.append(l_gate(site_qubits[j]).control(link_qubits, ctrl_state=control_state), controls + targets[j])
                dir += 1
            while dir > 0:
                qc.append(r_gate(site_qubits[j]).control(link_qubits, ctrl_state=control_state), controls + targets[j])
                dir -= 1
    return qc


def lcu_success_probability(state: Statevector, num_ancillas: int) -> float:
    """Success probability of the LCU: total amplitude in the |0...0⟩ subspace
    of the ancilla qubits (top num_ancillas qubits in Qiskit ordering)."""
    arr = np.array(state)
    subspace_size = len(arr) // (2 ** num_ancillas)
    return float(np.sum(np.abs(arr[:subspace_size]) ** 2))


def recover_quantity_quantum_macros(
    state: Statevector,
    site_dims: list[int],
    num_links: int,
    original_sum: np.float64,
) -> NDArray[np.float64]:
    state_array = np.array(state)
    num_sites = np.prod(site_dims)

    density = state_array[:num_sites].reshape(site_dims, order='F')
    density = np.real(density) / np.linalg.norm(density)
    return original_sum * (density / np.sum(density))
