# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Quantum Lattice Boltzmann Method (QLBM) implementation that encodes the classical LBM algorithm as a quantum circuit using Qiskit. The project compares quantum and classical LBM simulations across 1D, 2D, and 3D lattice configurations with support for boundary conditions.

## Environment Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS

# Install dependencies (uv.lock is present, prefer uv if available)
pip install -e .
```

Dependencies: `numpy`, `qiskit>=1.0`, `qiskit-aer`, `matplotlib`, `pandas`, `scipy`, `seaborn`, `ipympl`, `pylatexenc`.

## Running Experiments

All experiments are run through Jupyter notebooks. There is no standalone test suite.

```bash
jupyter notebook
```

Start from `ExperimentTemplate.ipynb` at the repo root. It uses `%autoreload 2` so library edits are picked up automatically without manual reloads.

Original experiment notebooks are archived under `old_experiments/`.

## Architecture

### Package: `qlbm/`

The library is a flat package. All simulation logic lives here.

```
qlbm/
  __init__.py           ← re-exports the full public API
  circuits.py           ← quantum gates and encoding
  physics.py            ← collision operator, BCs, SVD decomp
  simulate.py           ← simulate_flow, simulate_flow_classical
  analysis.py           ← save_rmse_comparison, write_array_to_csv
  initial_conditions.py ← initial density helpers (1D/2D/3D + BC helpers)
  visualization.py      ← snapshot / animation helpers
```

**Two simulation paths share the same config format:**

```python
config = [
    (iterations, velocity_field, links, weights, speed_of_sound, boundary_conditions),
    # ... additional phases
]
qlbm.simulate_flow(initial_density, config, "output.csv")           # quantum
qlbm.simulate_flow_classical(initial_density, config, "output.csv") # classical
```

- `velocity_field`: shape `(dimensions, *spatial_dims)` — macroscopic velocity at each site
- `links`: list of velocity vectors, e.g. `[[0], [-1], [1]]` for D1Q3
- `weights`: equilibrium distribution weights (must sum to 1)
- `boundary_conditions`: `None` or array of shape `(*spatial_dims, num_velocities, num_velocities)` — local linear transformation applied per-site before streaming

Results are written as CSV rows, one row per timestep (density field flattened column-major / Fortran order).

### Quantum Circuit Structure (per iteration)

1. **Encode** — amplitude-encode the density array into a `Statevector`; link qubits initialized to `|0⟩`
2. **encode_links** — Hadamard gates on link qubits to create uniform superposition over velocity directions
3. **collision** — applies the BGK collision operator:
   - Diagonal (no BC): `DiagonalGate` with complex entries `d ± i√(1-d²)`
   - Non-diagonal (with BC): SVD decomposition via `decompose_matrix_svd()`, manually constructed controlled unitaries
4. **propagation** — controlled cyclic shifts (`R`/`L` gates) on site qubits, conditioned on each link state
5. **macros** — SWAP+Hadamard sequence on link+ancilla qubits to project out the density

### Qubit Layout

```
[site qubits (per dim)] [link qubits] [ancilla]
```

- Site qubits: `ceil(log2(sites))` per spatial dimension
- Link qubits: `ceil(log2(num_links))`
- 1 ancilla qubit for the collision oracle

### Key Implementation Details

- **Padding**: velocities are padded to the next power of 2 (`2^link_qubits`); extra slots get zero collision weights
- **Circuit caching**: `simulate_flow()` pre-builds and transpiles all gates before the simulation loop, keyed by a hash of the combined collision+BC matrix and links
- **State recovery**: `recover_quantity_quantum_macros()` extracts density from the first `num_sites` amplitudes, renormalizes, and rescales to preserve total density
- **BC matrix construction**: `bc_config_to_matrix()` converts the per-site BC tensor into a block-structured matrix; `combine_collision_bc_matrices()` left-multiplies it onto the collision diagonal

### Experiment Output Convention

Results are saved under `experiments/<ExperimentName_YYYYMMDD_HHMMSS>/`:
- `classical.csv` / `quantum.csv` — per-iteration density snapshots
- `rmse.png` — RMSE between quantum and classical over time
- `snapshots.png` — side-by-side spatial snapshots

### Lattice Configurations Used in Experiments

| Name | Dims | Velocities | Links example |
|------|------|------------|---------------|
| D1Q3 | 1D | 3 | `[[-1],[0],[1]]` |
| D2Q5 | 2D | 5 | `[[0,0],[1,0],[-1,0],[0,1],[0,-1]]` |
| D2Q9 | 2D | 9 | rest + 4 axis + 4 diagonal |
| D3Q  | 3D | varies | — |
