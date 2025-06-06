# Quantum Lattice Boltzmann Method Implementation

## Requirements

- Python 3.8+
- Required packages:
  ```
  numpy>=1.24.0
  qiskit>=1.0.0
  qiskit-aer>=0.12.0
  matplotlib>=3.7.0
  pandas>=2.0.0
  ```

## Installation

1. Clone the repository:
   ```powershell
   git clone [your-repo-url]
   cd [repo-name]
   ```

2. Install dependencies:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

## Usage

### Running Simulations

You can run simulations using either the quantum or classical implementation. Check `Experiment.ipynb` for example usage:

```python
jupyter notebook Experiment.ipynb
```

Basic usage example:

```python
import numpy as np
from qlbmlib import simulate_flow, simulate_flow_classical

# Set up initial conditions
initial_density = np.ones((8, 8)) / 64  # 8x8 uniform distribution
velocity_field = np.zeros((2, 8, 8))    # 2D zero velocity field
links = [[0,0], [1,0], [-1,0], [0,1], [0,-1]]  # D2Q5 lattice
weights = [1/5] * 5
speed_of_sound = 1/np.sqrt(3)

# Configure simulation
config = [(100, velocity_field, links, weights, speed_of_sound)]

# Run quantum simulation
simulate_flow(initial_density, config, "quantum_results.csv", enable_quantum_macros=True)

# Run classical simulation
simulate_flow_classical(initial_density, config, "classical_results.csv")
```

### Comparing Results

To compare quantum and classical simulation results:

```python
from qlbmlib import save_rmse_comparison

save_rmse_comparison(
    "quantum_results.csv",
    "classical_results.csv",
    dimensions=(8, 8),
    output_path="rmse_plot.png",
    labels=("Quantum", "Classical")
)
```
