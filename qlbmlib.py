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
import pandas as pd
import matplotlib.pyplot as plt

def write_array_to_csv(array: NDArray[np.float64], filename: str, mode: str = 'w') -> None:
    """Write a numpy array to a CSV file.
    
    Args:
        array: Numpy array to write to file
        filename: Path to the CSV file
        mode: File opening mode ('w' for write/overwrite, 'a' for append)
    """
    # Flatten array in Fortran order to match simulation data format
    flat_array = array.flatten(order='F')
    
    with open(filename, mode, newline='') as file:
        writer = csv.writer(file)
        writer.writerow(flat_array)

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
        print(f"Link {i}: {link_vals}")
        
        # Reshape and add to total density
        density += link_vals.reshape(site_dims, order='F')
    
    density = np.real(density)
    return original_norm * density

def recover_quantity_quantum_macros(state: Statevector, site_dims: list[int], num_links: int, original_norm: np.float64) -> NDArray[np.float64]:
    # Get the statevector as numpy array
    state_array = np.array(state)

    # Calculate total number of sites
    num_sites = np.prod(site_dims)
    
    # Initialize density array
    density = state_array[:num_sites].reshape(site_dims, order='F')

    total_links = 2**np.ceil(np.log2(num_links))
    
    density = np.real(density)
    return original_norm * density * total_links

def get_renorm_coeff(num_velocities: int) -> NDArray[np.float64]:
    floor_qubits = int(np.floor(np.log2(num_velocities)))
    full_velocities = 2**floor_qubits

    coeffs = np.full(num_velocities, 2.0)
    for i in range(num_velocities - full_velocities):
        coeffs[i] *= np.sqrt(2)
        coeffs[-i - 1] *= np.sqrt(2)

    return coeffs

def get_collision_diagonal(link_qubits: int, links: list[list[int]], weights: list[float], velocity_field: NDArray[np.float64], speed_of_sound: np.float64):
    normalized_links = []
    for link in links:
        link_vector = np.array(link)
        link_norm = np.linalg.norm(link_vector)
        if link_norm > 0:  # Skip normalization for rest particle
            link_vector = link_vector / link_norm
        normalized_links.append(link_vector.tolist())
    
    num_links = len(links)

    dimension = list(velocity_field.shape)[0]
    velocity_field = velocity_field.reshape(dimension, -1, order='F')
    field_size = len(velocity_field[0])

    max_links = 2**link_qubits
    
    blocks = []
    for i, link in enumerate(normalized_links):
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

    norm = np.float64(np.linalg.norm(initial_density))

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
                initial_density = recover_quantity(state, sites_per_dim, num_links, norm)
                norm = np.float64(np.linalg.norm(initial_density))
                
                # Write current state to CSV
                writer.writerow(initial_density.flatten(order='F'))
                
                # Print detailed timing information
                print(f"Iteration {i+1}/{total_iterations}:")
                print(f"  Compilation: {compile_time:.3f} seconds")
                print(f"  Execution: {execute_time:.3f} seconds")
                print(f"  Total: {compile_time + execute_time:.3f} seconds")
            current_iterations += iterations

def simulate_flow_classical(initial_density: NDArray[np.float64],
                        configs: list[tuple[int, NDArray[np.float64], list[list[int]], list[float], float]],
                        filename: str) -> None:
    """Simulate classical Lattice Boltzmann Method flow.
    
    Args:
        initial_density: Initial density distribution
        configs: List of tuples containing (iterations, velocity_field, links, weights, speed_of_sound)
        filename: Path to save the CSV file with simulation results
    """
    # Convert links and weights to numpy arrays for easier handling
    total_iterations = np.sum(list(map(lambda c: c[0], configs)))
    density = initial_density.copy()
    
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(density.flatten(order='F'))
        
        current_iterations = 0
        for config in configs:
            iterations, velocity_field, links, weights, speed_of_sound = config
            print(f"Classical simulation: iterations {current_iterations}-{current_iterations + iterations}/{total_iterations}")
            
            dimension = list(velocity_field.shape)[0]
            
            # Evolution loop
            for t in range(iterations):
                print(f"Classical Iteration {current_iterations + t + 1}/{total_iterations}")
                # Calculate equilibrium distributions
                f = []
                for i in range(len(links)):
                    # Normalize link vector
                    link_vector = np.array(links[i])
                    link_norm = np.linalg.norm(link_vector)
                    if link_norm > 0:  # Skip normalization for rest particle
                        link_vector = link_vector / link_norm
                    
                    # Project velocity onto normalized lattice direction
                    v_proj = np.sum([
                        link_vector[d] * velocity_field[d]
                        for d in range(dimension)
                    ], axis=0)
                    
                    # Calculate equilibrium distribution
                    fi = weights[i] * density * (1 + v_proj / (speed_of_sound ** 2))
                    f.append(fi)
                
                # Streaming step
                for i in range(1, len(links)):  # Skip rest particle (index 0)
                    fi = f[i]
                    # Handle arbitrary velocity vectors by rolling multiple times if needed
                    for dim, shift in enumerate(links[i]):
                        if shift != 0:  # Only roll if there's movement in this direction
                            # Get number of cells to shift (supports vectors like [2,0] or [-2,1])
                            shift_amount = shift
                            # Roll the distribution the required number of times
                            fi = np.roll(fi, shift_amount, axis=dim)
                    f[i] = fi
                
                # Update density field
                density = np.sum(f, axis=0)
                
                # Save current state
                writer.writerow(density.flatten(order='F'))
                
            current_iterations += iterations
    
    print(f"Classical simulation complete. Results saved to {filename}")

def save_rmse_comparison(file1: str, file2: str, dimensions: tuple[int, ...], 
                        output_path: str, 
                        labels: tuple[str, str] = ("Quantum", "Classical")) -> None:
    """Save a plot showing RMSE evolution between two simulations.
    
    Args:
        file1: Path to first simulation CSV file
        file2: Path to second simulation CSV file
        dimensions: Tuple of dimensions for reshaping the data
        output_path: Path to save the figure
        labels: Tuple of labels for the two simulations
    """
    # Read both CSV files
    df1 = pd.read_csv(file1, header=None)
    df2 = pd.read_csv(file2, header=None)
    
    if len(df1) != len(df2):
        raise ValueError(f"Files have different number of iterations: {len(df1)} vs {len(df2)}")
    
    rmse_values = []
    iterations = list(range(len(df1)))
    
    for i in range(len(df1)):
        frame1 = df1.iloc[i].to_numpy().reshape(dimensions, order='F')
        frame2 = df2.iloc[i].to_numpy().reshape(dimensions, order='F')
        
        # Calculate regular RMSE
        rmse = np.sqrt(np.mean((frame1 - frame2)**2))
        rmse_values.append(rmse)
        
    
       # Create the figure
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, rmse_values, 'b-', linewidth=2)
    plt.grid(True)
    plt.xlabel('Iteration')
    plt.ylabel('RMSE')
    plt.title(f'RMSE Evolution: {labels[0]} vs {labels[1]}')
    
    plt.tight_layout()
    
    # Save the figure
    if output_path is None:
        os.makedirs("experiments/figures", exist_ok=True)
        output_path = f"experiments/figures/rmse_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"RMSE comparison saved to {output_path}")