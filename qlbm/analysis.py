import csv
import os
import numpy as np
from numpy.typing import NDArray
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile


def write_array_to_csv(array: NDArray[np.float64], filename: str, mode: str = 'w') -> None:
    flat_array = array.flatten(order='F')

    with open(filename, mode, newline='') as file:
        writer = csv.writer(file)
        writer.writerow(flat_array)


def save_rmse_comparison(
    file1: str,
    file2: str,
    dimensions: tuple[int, ...],
    output_path: str,
    labels: tuple[str, str] = ("Quantum", "Classical"),
) -> None:
    df1 = pd.read_csv(file1, header=None)
    df2 = pd.read_csv(file2, header=None)

    if len(df1) != len(df2):
        raise ValueError(f"Files have different number of iterations: {len(df1)} vs {len(df2)}")

    rmse_values = []
    iterations = list(range(len(df1)))

    for i in range(len(df1)):
        frame1 = df1.iloc[i].to_numpy().reshape(dimensions, order='F')
        frame2 = df2.iloc[i].to_numpy().reshape(dimensions, order='F')

        rmse = np.sqrt(np.mean((frame1 - frame2)**2))
        rmse_values.append(rmse)

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, rmse_values, 'b-', linewidth=2)
    plt.grid(True)
    plt.xlabel('Iteration')
    plt.ylabel('RMSE')
    plt.title(f'RMSE Evolution: {labels[0]} vs {labels[1]}')

    plt.tight_layout()

    if output_path is None:
        os.makedirs("experiments/figures", exist_ok=True)
        output_path = f"experiments/figures/rmse_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"RMSE comparison saved to {output_path}")


def count_gates(circuit: QuantumCircuit) -> dict[str, int]:
    """Count gates by type in a transpiled circuit (excludes reset instructions)."""
    counts = {}
    for instruction in circuit.data:
        name = instruction.operation.name
        if name == 'reset':
            continue
        counts[name] = counts.get(name, 0) + 1
    return counts


def transpile_stage(
    label: str,
    gate,
    num_qubits: int,
    qubit_indices: list[int],
    basis_gates: list[str] | None = None,
) -> dict:
    """Transpile a single stage and return gate counts."""
    qc = QuantumCircuit(num_qubits)
    qc.append(gate, qubit_indices)
    transpiled = transpile(qc, basis_gates=basis_gates)

    gates = count_gates(transpiled)
    return {
        "stage": label,
        "total_gates": sum(gates.values()),
        "depth": transpiled.depth(),
        "gate_breakdown": gates,
    }
