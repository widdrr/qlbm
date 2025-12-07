import numpy as np
from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Gate
from qiskit.circuit.library import HGate, DiagonalGate, UnitaryGate
from numpy.typing import NDArray
from qiskit_aer import StatevectorSimulator
from typing import Callable, Optional
import csv
import time
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.linalg import svd
from scipy.linalg import svd

def decompose_matrix_svd(A: NDArray[np.float64]) -> tuple[NDArray[np.complex128], NDArray[np.complex128], float]:
    A = A / np.linalg.norm(A, ord=2)
    
    V, singular_values, W_dagger = svd(A)
    
    # Ensure no calculation under sqrt(1 - (sigma_prime)^2) results in 
    # a negative number due to precision errors. Clip to 1.0.
    argument = 1.0 - np.minimum(singular_values**2, 1.0)
    
    # Lambda_diag is the diagonal of the matrix Lambda
    Lambda_diag = singular_values + 1j * np.sqrt(argument)
    
    # Construct the full diagonal matrix Lambda
    Lambda = np.diag(Lambda_diag)
    
    # U1 = V @ Lambda @ W_dagger
    U1 = V @ Lambda @ W_dagger
    
    # U2 = V @ Lambda_dagger @ W_dagger
    U2 = V @ Lambda.conj().T @ W_dagger
    
    return U1, U2

def write_array_to_csv(array: NDArray[np.float64], filename: str, mode: str = 'w') -> None:
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

def bc_config_to_matrix(bc: NDArray[np.float64], 
                        grid_size: int, 
                        num_velocities: int) -> NDArray[np.float64]:
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
        
        # Pad with identity for extra velocities
        for v in range(actual_num_velocities, num_velocities):
            idx = v * grid_size + site
            bc_matrix[idx, idx] = 1.0
    
    return bc_matrix.toarray()

def combine_collision_bc_matrices(collision_diagonal: NDArray[np.float64],
                                 bc_matrix: Optional[NDArray[np.float64]]) -> NDArray[np.float64]:
    if bc_matrix is None:
        return collision_diagonal
    
    collision_matrix_full = np.diag(collision_diagonal)
    combined = bc_matrix @ collision_matrix_full
    
    return combined

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
    qc = QuantumCircuit(link_qubits)

    for i in range(link_qubits):
        qc.h(i)

    return qc

def recover_quantity_quantum_macros(state: Statevector, site_dims: list[int], num_links: int, original_sum: np.float64) -> NDArray[np.float64]:
    state_array = np.array(state)
    num_sites = np.prod(site_dims)
    
    # rescale to preserve total quantity
    density = state_array[:num_sites].reshape(site_dims, order='F')
    density = np.real(density) / np.linalg.norm(density)
    return original_sum * (density / np.sum(density))

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

    if collision_matrix.ndim == 1:
        diagonal = collision_matrix
        unitary_1 = list(diagonal + 1.j*np.sqrt(1 - np.square(diagonal)))
        unitary_2 = list(diagonal - 1.j*np.sqrt(1 - np.square(diagonal)))

        qc.h(ancilla)
        qc.append(DiagonalGate(unitary_1).control(ctrl_state='0'), [ancilla] + target_qubits)
        qc.append(DiagonalGate(unitary_2).control(ctrl_state='1'), [ancilla] + target_qubits)
        qc.h(ancilla)
    else:
        print("Using SVD decomposition for non-diagonal collision matrix")
        # Use SVD decomposition for boundary condition matrices
        U1, U2 = decompose_matrix_svd(collision_matrix)
        print("Decomposition complete")

        # Manually construct controlled gates for performance
        # Using Kronecker product: U ⊗ |ctrl_state⟩⟨ctrl_state| + I ⊗ |1-ctrl_state⟩⟨1-ctrl_state|
        size = U1.shape[0]
        identity = np.eye(size, dtype=complex)
        proj_0 = np.array([[1, 0], [0, 0]], dtype=complex)
        proj_1 = np.array([[0, 0], [0, 1]], dtype=complex)
        
        # For ctrl_state='0': U ⊗ |0⟩⟨0| + I ⊗ |1⟩⟨1|
        controlled_U1 = np.kron(U1, proj_0) + np.kron(identity, proj_1)
        ctrl_gate1 = UnitaryGate(controlled_U1)
        
        # For ctrl_state='1': U ⊗ |1⟩⟨1| + I ⊗ |0⟩⟨0|
        controlled_U2 = np.kron(U2, proj_1) + np.kron(identity, proj_0)
        ctrl_gate2 = UnitaryGate(controlled_U2)

        qc.h(ancilla)
        qc.append(ctrl_gate1, [ancilla] + target_qubits)
        qc.append(ctrl_gate2, [ancilla] + target_qubits)
        qc.h(ancilla)
        print("Added manually-constructed controlled unitaries to circuit")

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

def simulate_flow(initial_density: NDArray[np.float64],
                  configs: list[tuple[int, NDArray[np.float64], list[list[int]], list[float], np.float64, Optional[NDArray[np.float64]]]],
                  filename: str) -> None:

    simulator = StatevectorSimulator()

    sites_per_dim = list(initial_density.shape)
    site_qubits_per_dim = [int(np.ceil(np.log2(sites))) for sites in sites_per_dim]
    site_qubits = np.sum(site_qubits_per_dim)

    original_sum = np.float64(np.sum(initial_density))

    total_iterations = np.sum(list(map(lambda c: c[0], configs)))

    print(f"Pre-building all circuit components for {len(configs)} configuration(s)...")
    
    # pre-build all circuit components for all configurations
    component_cache = {}  # Maps config_hash -> (encode_links, collision, propagation, macros, metadata)
    config_metadata = []  # List of (config_index, iterations, config_hash, num_links, link_qubits, recover_fn)
    
    for config_idx, config in enumerate(configs):
        (iterations, velocity_field, links, weights, speed_of_sound, boundary_conditions) = config
        
        num_links = len(links)
        link_qubits = int(np.ceil(np.log2(num_links)))
        collision_matrix = get_collision_diagonal(link_qubits, links, weights, velocity_field, speed_of_sound)
        
        # convert BC config
        bc_matrix = None
        if boundary_conditions is not None:
            grid_size = np.prod(sites_per_dim)
            padded_num_velocities = 2**link_qubits
            bc_matrix = bc_config_to_matrix(boundary_conditions, grid_size, padded_num_velocities)
        
        #handle caching
        combined_matrix = combine_collision_bc_matrices(collision_matrix, bc_matrix)
        config_parts = [
            combined_matrix.tobytes(),
            str(links).encode(),
        ]
        config_hash = hash(tuple(config_parts))
        config_metadata.append((config_idx, iterations, config_hash, num_links, link_qubits))
        
        # Build components if not already cached
        if config_hash not in component_cache:
            print(f"  Building components for configuration {config_idx + 1}/{len(configs)}...")
            
            num_qubits = site_qubits + link_qubits + 1
            
            # Encode links gate
            encode_links_gate = encode_links(link_qubits, num_links).to_gate(label='encode_links')
            qc_temp = QuantumCircuit(num_qubits)
            qc_temp.append(encode_links_gate, list(range(site_qubits, num_qubits - 1)))
            cached_encode_links = transpile(qc_temp, simulator)
            
            # Collision gate
            collision_gate = collision_nonuniform(site_qubits, link_qubits, combined_matrix).to_gate(label='collision')
            qc_temp = QuantumCircuit(num_qubits)
            qc_temp.append(collision_gate, list(range(num_qubits)))
            cached_collision = transpile(qc_temp, simulator)
            
            # Propagation gate
            propagation_gate = propagation(site_qubits_per_dim, link_qubits, links).to_gate(label='propagation')
            qc_temp = QuantumCircuit(num_qubits)
            qc_temp.append(propagation_gate, list(range(0, num_qubits - 1)))
            cached_propagation = transpile(qc_temp, simulator)

            # Macros gate
            macros_gate = macros(link_qubits).to_gate(label='macros')
            qc_temp = QuantumCircuit(num_qubits)
            qc_temp.append(macros_gate, list(range(site_qubits, num_qubits)))
            cached_macros = transpile(qc_temp, simulator)
            
            component_cache[config_hash] = (cached_encode_links, cached_collision, cached_propagation, cached_macros)
            print(f"    Components built and transpiled.")
        else:
            print(f"  Configuration {config_idx + 1}/{len(configs)} reuses components from previous configuration.")
    
    print(f"Pre-building complete. {len(component_cache)} unique component set(s) created.")
    print("Starting simulation...")
    
    current_iterations = 0
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(initial_density.flatten(order='F'))
        
        for config_idx, iterations, config_hash, num_links, link_qubits in config_metadata:
            print(f"\nConfiguration {config_idx + 1}/{len(configs)}: iterations {current_iterations + 1}-{current_iterations + iterations}/{total_iterations}")
            
            # cached components
            cached_encode_links, cached_collision, cached_propagation, cached_macros = component_cache[config_hash]

            # main loop
            for i in range(current_iterations, current_iterations + iterations):
                # Evolve state
                print(f"Iteration {i+1} running...")
                
                # encode state (must be done each iteration as density changes)
                state = encode(initial_density, link_qubits)
                if state.num_qubits is None:
                    raise ValueError("Failed to initialize quantum state")
                
                num_qubits = state.num_qubits
                qc = QuantumCircuit(num_qubits)
                qc.initialize(state)
                
                qc.compose(cached_encode_links, inplace=True)
                qc.compose(cached_collision, inplace=True)
                qc.compose(cached_propagation, inplace=True)
                if cached_macros is not None:
                    qc.compose(cached_macros, inplace=True)
                
                result = simulator.run(qc).result()
                state = result.get_statevector()
                
                initial_density = recover_quantity_quantum_macros(state, sites_per_dim, num_links, original_sum)
                writer.writerow(initial_density.flatten(order='F'))
            current_iterations += iterations

def simulate_flow_classical(initial_density: NDArray[np.float64],
                        configs: list[tuple[int, NDArray[np.float64], list[list[int]], list[float], float, NDArray[np.float64] | None]],
                        filename: str) -> None:
    # Convert links and weights to numpy arrays for easier handling
    total_iterations = np.sum(list(map(lambda c: c[0], configs)))
    density = initial_density.copy()
    
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(density.flatten(order='F'))
        
        current_iterations = 0
        for config in configs:
            iterations, velocity_field, links, weights, speed_of_sound, boundary_conditions = config
            print(f"Classical simulation: iterations {current_iterations}-{current_iterations + iterations}/{total_iterations}")
            
            dimension = list(velocity_field.shape)[0]
            num_velocities = len(links)
            
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
                
                if boundary_conditions is not None:
                    # Apply boundary conditions BEFORE streaming (linear transformation of velocity distributions at each site)
                    # f is a list of arrays, each of shape (*spatial_dims)
                    # We need to transform them into a tensor of shape (*spatial_dims, num_velocities)
                    # then apply the boundary condition matrices
                    f_stacked = np.stack(f, axis=-1)  # Shape: (*spatial_dims, num_velocities)
                    
                    # Apply the boundary condition transformation at each site
                    # boundary_conditions has shape (*spatial_dims, num_velocities, num_velocities)
                    # We want to compute: f_new[site, j] = sum_i(boundary_conditions[site, j, i] * f_old[site, i])
                    f_transformed = np.einsum('...ji,...i->...j', boundary_conditions, f_stacked)
                
                    # Unstack back into list of arrays
                    f = [f_transformed[..., i] for i in range(num_velocities)]
                
                # Streaming step
                for i in range(len(links)):
                    # Skip if this is a rest particle (all velocity components are zero)
                    if all(v == 0 for v in links[i]):
                        continue
                    
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