import numpy as np
from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Gate
from qiskit.circuit.library import HGate, DiagonalGate
from numpy.typing import NDArray
from qiskit_aer import StatevectorSimulator
from typing import Callable
import csv
import time
from datetime import datetime
import os

# Global debug file setup
debug_dir = 'experiments/debug'
os.makedirs(debug_dir, exist_ok=True)
debug_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
debug_file = os.path.join(debug_dir, f'deb\ug_{debug_timestamp}.log')
debug_array_dir = os.path.join(debug_dir, f'arrays_{debug_timestamp}')
os.makedirs(debug_array_dir, exist_ok=True)

def write_debug(message: str | NDArray, label: str = "") -> None:
    """Write a debug message or array to the debug files with timestamp.
    
    Args:
        message: The message to write (string) or array to save
        label: Optional label to identify the array in the filename
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    
    if isinstance(message, np.ndarray):
        # Save array to a separate npy file
        safe_label = "".join(c if c.isalnum() else "_" for c in label)
        array_file = os.path.join(debug_array_dir, f'{safe_label}_{len(os.listdir(debug_array_dir)):04d}.npy')
        np.save(array_file, message)
        
        # Write reference to the array file in the log
        with open(debug_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] Array saved ({message.shape}, {message.dtype}): {array_file}\n")
    else:
        # Write string message to log file
        with open(debug_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {message}\n")

def encode(variable: NDArray[np.float64], link_qubits: int) -> Statevector:

    state = variable.flatten(order='F')
    # Check if length is a power of 2
    if not (len(state) & (len(state) - 1) == 0):
        raise ValueError("Length of flattened array must be a power of 2")

    state = Statevector(state / np.linalg.norm(state))
    ancilla = Statevector([1.0, 0.0])
    for _ in range(link_qubits):
        state = state.expand(ancilla)

    
    return state.expand(ancilla)

def get_ctrl_qubits(binary_str: str)-> tuple[list[int], int, str]:
    length = len(binary_str)
    for i, bit in enumerate(binary_str):
        if i == length - 1:
            raise ValueError("Invalid binary representation")
        if bit == '1':
            return (list(range(length - i - 1)), length - i - 1, binary_str[i+1:])
    raise ValueError("Invalid binary representation")

def encode_links(link_qubits: int, num_links: int) -> QuantumCircuit:
    floor_qubits = int(np.floor(np.log2(num_links)))
    floor_links = 2**floor_qubits
    qc = QuantumCircuit(link_qubits)

    for i in range(link_qubits):
        qc.h(i)

    return qc

def recover_quantity_classical_macros(state: Statevector, site_dims: list[int], num_links: int, original_norm: np.float64) -> NDArray[np.float64]:
    # Get the statevector as numpy array
    state_array = np.array(state)
    
    # Calculate total number of sites
    num_sites = np.prod(site_dims)
    
    # Initialize density array
    density = np.zeros(site_dims, dtype=complex)
    
    # Sum up contributions from each link direction
    for i in range(num_links):
        # Extract values for this link direction
        start_idx = i * num_sites
        end_idx = (i + 1) * num_sites
        link_vals = state_array[start_idx:end_idx]
        
        # Reshape and add to total density
        density += link_vals.reshape(site_dims, order='F')
    
    density = np.real(density)
    return original_norm * density / np.linalg.norm(density)

def recover_quantity_quantum_macros(state: Statevector, site_dims: list[int], num_links: int, original_norm: np.float64) -> NDArray[np.float64]:
    # Get the statevector as numpy array
    state_array = np.array(state)
    
    # Calculate total number of sites
    num_sites = np.prod(site_dims)
    
    # Initialize density array
    density = state_array[:num_sites].reshape(site_dims, order='F')
    
    density = np.real(density)
    return original_norm * density / np.linalg.norm(density)

def get_renorm_coeff(num_velocities: int) -> NDArray[np.float64]:
    floor_qubits = int(np.floor(np.log2(num_velocities)))
    full_velocities = 2**floor_qubits

    coeffs = np.full(num_velocities, 2.0)
    for i in range(num_velocities - full_velocities):
        coeffs[i] *= np.sqrt(2)
        coeffs[-i - 1] *= np.sqrt(2)

    return coeffs

def get_collision_diagonal(link_qubits: int, links: list[list[int]], weights: list[float], velocity_field: NDArray[np.float64], speed_of_sound: np.float64):
    num_links = len(links)
    
    dimension = list(velocity_field.shape)[0]
    velocity_field = velocity_field.reshape(dimension, -1, order='F')
    field_size = len(velocity_field[0])

    max_links = 2**link_qubits
    
    blocks = []
    for i, link in enumerate(links):
        link_velocity = np.sum([(link[d]*velocity_field[d]) for d in range(dimension)], axis=0)
        block = weights[i] * (1 + link_velocity / (speed_of_sound ** 2))
        blocks.append(block)

    for _ in range(num_links, max_links):
        blocks.append(np.zeros(field_size))


    matrix = np.concatenate(blocks)
        
    return matrix

def collision_nonuniform(site_qubits: int, link_qubits: int, collision_matrix: NDArray[np.float64]) -> QuantumCircuit:
    num_qubits = site_qubits + link_qubits + 1
    qc = QuantumCircuit(num_qubits)
    
    ancilla = num_qubits - 1
    target_qubits = list(range(ancilla))

    unitary_1 = list(collision_matrix + 1.j*np.sqrt(1 - np.square(collision_matrix)))
    unitary_2 = list(collision_matrix - 1.j*np.sqrt(1 - np.square(collision_matrix)))

    qc.h(ancilla)
    qc.append(DiagonalGate(unitary_1).control(ctrl_state='0'), [ancilla] + target_qubits)
    qc.append(DiagonalGate(unitary_2).control(ctrl_state='1'), [ancilla] + target_qubits)
    qc.h(ancilla)

    return qc

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

def propagation(site_qubits: list[int], link_qubits: int, links: list[list[int]]) -> QuantumCircuit:
    
    tot_site_qubits = np.sum(site_qubits)
    num_qubits = tot_site_qubits + link_qubits
    qc = QuantumCircuit(num_qubits)
    
    targets = [list(range(sum(site_qubits[:i]), sum(site_qubits[:i+1]))) for i in range(len(site_qubits))]
    targets = [list(range(sum(site_qubits[:i]), sum(site_qubits[:i+1]))) for i in range(len(site_qubits))]
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
    for i, link, in enumerate(links):
        control_state = np.binary_repr(i, link_qubits)
        for j, dir in enumerate(link):
            #perform left streaming 'dir' cells away
            while dir < 0:
                qc.append(l_gate(site_qubits[j]).control(link_qubits, ctrl_state=control_state), controls + targets[j])
                dir += 1
            #perform right streaming 'dir' cells away
            while dir > 0:
                qc.append(r_gate(site_qubits[j]).control(link_qubits, ctrl_state=control_state), controls + targets[j])
                dir -= 1
    return qc

def macros(link_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(link_qubits + 1)

    ancilla = link_qubits
    
    for i in range(link_qubits):
        qc.swap(i,ancilla)
        qc.h(ancilla)
    
    return qc

def create_circuit(density: NDArray[np.float64], 
                   site_qubits: int,
                   site_qubits_per_dim: list[int],
                   link_qubits: int,
                   num_links: int,
                   links: list[list[int]],
                   collision_matrix: NDArray[np.float64], 
                   enable_quantum_macros: bool)-> tuple[QuantumCircuit, Callable]:
    
    # Initialize state
    state = encode(density, link_qubits)
    if state.num_qubits is None:
        raise ValueError("Failed to initialize quantum state")

    qc = QuantumCircuit(state.num_qubits)
    qc.initialize(state)
    qc.append(encode_links(link_qubits, num_links), list(range(site_qubits, state.num_qubits -1)))
    qc.append(collision_nonuniform(site_qubits, link_qubits, collision_matrix), list(range(state.num_qubits)))
    qc.append(propagation(site_qubits_per_dim, link_qubits, links), list(range(0, state.num_qubits - 1)))

    recover_quantity = recover_quantity_classical_macros
    if enable_quantum_macros:
        qc.append(macros(link_qubits), list(range(site_qubits, state.num_qubits)))
        qc.barrier()
        recover_quantity = recover_quantity_quantum_macros

    return (qc, recover_quantity)

def simulate_flow(initial_density: NDArray[np.float64],
                  configs: list[tuple[int, NDArray[np.float64], list[list[int]], list[float], np.float64]],
                  filename: str,
                  enable_quantum_macros: bool) -> None:

    simulator = StatevectorSimulator()

    sites_per_dim = list(initial_density.shape)
    site_qubits_per_dim = [int(np.ceil(np.log2(sites))) for sites in sites_per_dim]
    site_qubits = np.sum(site_qubits_per_dim)

    # Get original norm for proper normalization
    original_norm = np.float64(np.linalg.norm(initial_density))

    total_iterations = np.sum(list(map(lambda c: c[0], configs)))

    current_iterations = 0
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        # Write initial state as first row
        writer.writerow(initial_density.flatten(order='F'))
        
        for config in configs:
            (iterations, velocity_field, links, weights, speed_of_sound) = config
            print(f"Circuit configuration: will run iterations {current_iterations}-{current_iterations + iterations}/{total_iterations} with this configuration")

            num_links = len(links)
            link_qubits = int(np.ceil(np.log2(num_links)))
            collision_matrix = get_collision_diagonal(link_qubits, links, weights, velocity_field, speed_of_sound)

            # Evolution loop
            for i in range(current_iterations, current_iterations + iterations):
                # Evolve state
                print(f"Iteration {i+1} running...")
                
                compile_start = time.time()
                qc, recover_quantity = create_circuit(initial_density, site_qubits, site_qubits_per_dim, link_qubits, num_links, links, collision_matrix, enable_quantum_macros)
                qc = transpile(qc, simulator)
                compile_time = time.time() - compile_start

                # Measure execution time
                execute_start = time.time()
                result = simulator.run(qc).result()
                state = result.get_statevector()
                execute_time = time.time() - execute_start
                
                # Recover density and normalize
                initial_density = recover_quantity(state, sites_per_dim, num_links, original_norm)
                
                # Write current state to CSV
                writer.writerow(initial_density.flatten(order='F'))
                
                # Print detailed timing information
                print(f"Iteration {i+1}/{total_iterations}:")
                print(f"  Compilation: {compile_time:.3f} seconds")
                print(f"  Execution: {execute_time:.3f} seconds")
                print(f"  Total: {compile_time + execute_time:.3f} seconds")
            current_iterations += iterations