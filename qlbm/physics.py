import numpy as np
from numpy.typing import NDArray
from typing import Optional
from scipy.linalg import svd
from scipy.sparse import lil_matrix
from qiskit import QuantumCircuit
from qiskit.circuit.library import DiagonalGate, UnitaryGate


def decompose_matrix_svd(A: NDArray[np.float64]) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    A = A / np.linalg.norm(A, ord=2)

    V, singular_values, W_dagger = svd(A)

    # Ensure no calculation under sqrt(1 - (sigma_prime)^2) results in
    # a negative number due to precision errors. Clip to 1.0.
    argument = 1.0 - np.minimum(singular_values**2, 1.0)

    Lambda_diag = singular_values + 1j * np.sqrt(argument)
    Lambda = np.diag(Lambda_diag)

    U1 = V @ Lambda @ W_dagger
    U2 = V @ Lambda.conj().T @ W_dagger

    return U1, U2


def get_collision_diagonal(
    link_qubits: int,
    links: list[list[int]],
    weights: list[float],
    velocity_field: NDArray[np.float64],
    speed_of_sound: np.float64,
) -> NDArray[np.float64]:
    normalized_links = []
    for link in links:
        link_vector = np.array(link)
        link_norm = np.linalg.norm(link_vector)
        if link_norm > 0:
            link_vector = link_vector / link_norm
        normalized_links.append(link_vector.tolist())

    num_links = len(links)
    dimension = list(velocity_field.shape)[0]
    velocity_field = velocity_field.reshape(dimension, -1, order='F')
    field_size = len(velocity_field[0])

    max_links = 2**link_qubits

    blocks = []
    for i, link in enumerate(normalized_links):
        link_velocity = np.sum([(link[d] * velocity_field[d]) for d in range(dimension)], axis=0)
        block = weights[i] * (1 + link_velocity / (speed_of_sound**2))
        blocks.append(block)

    for _ in range(num_links, max_links):
        blocks.append(np.zeros(field_size))

    return np.concatenate(blocks)


def bc_config_to_matrix(
    bc: NDArray[np.float64],
    grid_size: int,
    num_velocities: int,
) -> NDArray[np.float64]:
    actual_num_velocities = bc.shape[1]
    total_size = grid_size * num_velocities
    bc_matrix = lil_matrix((total_size, total_size))

    for site in range(grid_size):
        for j in range(actual_num_velocities):
            for i in range(actual_num_velocities):
                if bc[site, j, i] != 0:
                    row = j * grid_size + site
                    col = i * grid_size + site
                    bc_matrix[row, col] = bc[site, j, i]

        for v in range(actual_num_velocities, num_velocities):
            idx = v * grid_size + site
            bc_matrix[idx, idx] = 1.0

    return bc_matrix.toarray()


def combine_collision_bc_matrices(
    collision_diagonal: NDArray[np.float64],
    bc_matrix: Optional[NDArray[np.float64]],
) -> NDArray[np.float64]:
    if bc_matrix is None:
        return collision_diagonal

    collision_matrix_full = np.diag(collision_diagonal)
    return bc_matrix @ collision_matrix_full


def collision_nonuniform(
    site_qubits: int,
    link_qubits: int,
    collision_matrix: NDArray[np.float64],
) -> QuantumCircuit:
    num_qubits = site_qubits + link_qubits + 1
    qc = QuantumCircuit(num_qubits)

    ancilla = num_qubits - 1
    target_qubits = list(range(ancilla))

    if collision_matrix.ndim == 1:
        diagonal = collision_matrix
        unitary_1 = list(diagonal + 1.j * np.sqrt(1 - np.square(diagonal)))
        unitary_2 = list(diagonal - 1.j * np.sqrt(1 - np.square(diagonal)))

        qc.h(ancilla)
        qc.append(DiagonalGate(unitary_1).control(ctrl_state='0'), [ancilla] + target_qubits)
        qc.append(DiagonalGate(unitary_2).control(ctrl_state='1'), [ancilla] + target_qubits)
        qc.h(ancilla)
    else:
        print("Using SVD decomposition for non-diagonal collision matrix")
        U1, U2 = decompose_matrix_svd(collision_matrix)
        print("Decomposition complete")

        size = U1.shape[0]
        identity = np.eye(size, dtype=complex)
        proj_0 = np.array([[1, 0], [0, 0]], dtype=complex)
        proj_1 = np.array([[0, 0], [0, 1]], dtype=complex)

        controlled_U1 = np.kron(U1, proj_0) + np.kron(identity, proj_1)
        ctrl_gate1 = UnitaryGate(controlled_U1)

        controlled_U2 = np.kron(U2, proj_1) + np.kron(identity, proj_0)
        ctrl_gate2 = UnitaryGate(controlled_U2)

        qc.h(ancilla)
        qc.append(ctrl_gate1, [ancilla] + target_qubits)
        qc.append(ctrl_gate2, [ancilla] + target_qubits)
        qc.h(ancilla)
        print("Added manually-constructed controlled unitaries to circuit")

    return qc
