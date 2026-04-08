import csv
import gc
import numpy as np
from numpy.typing import NDArray
from typing import Optional
from qiskit import QuantumCircuit, transpile
from qiskit_aer import StatevectorSimulator

from .circuits import encode, encode_links, propagation, macros, recover_quantity_quantum_macros
from .physics import get_collision_diagonal, bc_config_to_matrix, combine_collision_bc_matrices, collision_nonuniform


def simulate_flow(
    initial_density: NDArray[np.float64],
    configs: list[tuple[int, NDArray[np.float64], list[list[int]], list[float], np.float64, Optional[NDArray[np.float64]]]],
    filename: str,
) -> None:
    simulator = StatevectorSimulator()

    sites_per_dim = list(initial_density.shape)
    site_qubits_per_dim = [int(np.ceil(np.log2(sites))) for sites in sites_per_dim]
    site_qubits = np.sum(site_qubits_per_dim)

    original_sum = np.float64(np.sum(initial_density))

    total_iterations = np.sum(list(map(lambda c: c[0], configs)))

    print(f"Pre-building all circuit components for {len(configs)} configuration(s)...")

    component_cache = {}
    config_metadata = []

    for config_idx, config in enumerate(configs):
        (iterations, velocity_field, links, weights, speed_of_sound, boundary_conditions) = config

        num_links = len(links)
        link_qubits = int(np.ceil(np.log2(num_links)))
        collision_matrix = get_collision_diagonal(link_qubits, links, weights, velocity_field, speed_of_sound)

        bc_matrix = None
        if boundary_conditions is not None:
            grid_size = np.prod(sites_per_dim)
            padded_num_velocities = 2**link_qubits
            nv_actual = boundary_conditions.shape[-1]
            ndim_spatial = boundary_conditions.ndim - 2
            if ndim_spatial > 1:
                # Flatten spatial dimensions in Fortran order (x varies fastest)
                perm = list(reversed(range(ndim_spatial))) + [ndim_spatial, ndim_spatial + 1]
                bc_flat = np.ascontiguousarray(
                    np.transpose(boundary_conditions, perm)
                ).reshape(grid_size, nv_actual, nv_actual)
            else:
                bc_flat = boundary_conditions
            bc_matrix = bc_config_to_matrix(bc_flat, grid_size, padded_num_velocities)

        combined_matrix = combine_collision_bc_matrices(collision_matrix, bc_matrix)
        config_parts = [
            combined_matrix.tobytes(),
            str(links).encode(),
        ]
        config_hash = hash(tuple(config_parts))
        config_metadata.append((config_idx, iterations, config_hash, num_links, link_qubits))

        if config_hash not in component_cache:
            print(f"  Building components for configuration {config_idx + 1}/{len(configs)}...")

            num_qubits = site_qubits + link_qubits + 1

            encode_links_gate = encode_links(link_qubits, num_links).to_gate(label='encode_links')
            qc_temp = QuantumCircuit(num_qubits)
            qc_temp.append(encode_links_gate, list(range(site_qubits, num_qubits - 1)))
            cached_encode_links = transpile(qc_temp, simulator)

            collision_gate = collision_nonuniform(site_qubits, link_qubits, combined_matrix).to_gate(label='collision')
            qc_temp = QuantumCircuit(num_qubits)
            qc_temp.append(collision_gate, list(range(num_qubits)))
            cached_collision = transpile(qc_temp, simulator)

            propagation_gate = propagation(site_qubits_per_dim, link_qubits, links).to_gate(label='propagation')
            qc_temp = QuantumCircuit(num_qubits)
            qc_temp.append(propagation_gate, list(range(0, num_qubits - 1)))
            cached_propagation = transpile(qc_temp, simulator)

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

            cached_encode_links, cached_collision, cached_propagation, cached_macros = component_cache[config_hash]

            for i in range(current_iterations, current_iterations + iterations):
                print(f"Iteration {i + 1} running...")

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

                job = simulator.run(qc)
                result = job.result()
                state = result.get_statevector()
                del job, result, qc
                gc.collect()

                initial_density = recover_quantity_quantum_macros(state, sites_per_dim, num_links, original_sum)
                writer.writerow(initial_density.flatten(order='F'))
            current_iterations += iterations


def simulate_flow_classical(
    initial_density: NDArray[np.float64],
    configs: list[tuple[int, NDArray[np.float64], list[list[int]], list[float], float, NDArray[np.float64] | None]],
    filename: str,
) -> None:
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

            for t in range(iterations):
                print(f"Classical Iteration {current_iterations + t + 1}/{total_iterations}")
                f = []
                for i in range(len(links)):
                    link_vector = np.array(links[i])
                    link_norm = np.linalg.norm(link_vector)
                    if link_norm > 0:
                        link_vector = link_vector / link_norm

                    v_proj = np.sum([
                        link_vector[d] * velocity_field[d]
                        for d in range(dimension)
                    ], axis=0)

                    fi = weights[i] * density * (1 + v_proj / (speed_of_sound**2))
                    f.append(fi)

                if boundary_conditions is not None:
                    f_stacked = np.stack(f, axis=-1)
                    f_transformed = np.einsum('...ji,...i->...j', boundary_conditions, f_stacked)
                    f = [f_transformed[..., i] for i in range(num_velocities)]

                for i in range(len(links)):
                    if all(v == 0 for v in links[i]):
                        continue

                    fi = f[i]
                    for dim, shift in enumerate(links[i]):
                        if shift != 0:
                            fi = np.roll(fi, shift, axis=dim)
                    f[i] = fi

                density = np.sum(f, axis=0)
                writer.writerow(density.flatten(order='F'))

            current_iterations += iterations

    print(f"Classical simulation complete. Results saved to {filename}")
