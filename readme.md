# Radio SLAM Observability Benchmarking Suite

## 1. Project Overview
This repository provides a high-fidelity simulation framework to evaluate the performance gains of **Map-assisted Radio SLAM** compared to traditional **Mapless Radio SLAM** . The primary goal is to verify how prior geometric constraints (like floorplans) improve system observability and estimation precision.

The framework compares two paradigms:
* **Mapless Model**: Treats reflected multipath signals as independent virtual anchors (VAs) with unknown positions.
* **Map-Assisted Model**: Utilizes known wall geometry to constrain virtual anchors as mirrored reflections of the physical anchor. Every reflected multipath acts as an additional virtual ranging measurement, enhancing the observability of the anchor's state.

## 2. Project Architecture
The repository is structured for modularity, allowing developers to easily swap out map geometries, EKF solvers, and initial prior configurations.

    radio-slam-comparison/
    ├── src/
    │   ├── core/
    │   │   ├── dynamics.py       # Velocity Random-Walk & 2-state Clock models
    │   │   ├── map_manager.py    # Wall geometries & Householder M_i matrices
    │   │   ├── measurement.py    # Jacobian (H) calculation for Mapless/Map-assisted
    │   │   └── observability.py  # l-step matrix builder & SVD rank analysis
    │   ├── estimators/
    │   │   ├── ekf_base.py       # Base EKF with robust Cholesky/SVD solvers
    │   │   ├── mapless_ekf.py    # State-augmented EKF (10 + 2N dimensions)
    │   │   └── map_aided_ekf.py  # Map-constrained EKF (10 dimensions)
    │   ├── experiments/
    │   │   ├── config.py         # Definitions for Cases 1-4 (Initial Priors)
    │   │   └── runner.py         # Monte Carlo engine for 50-run validation
    │   └── utils/
    │       ├── stats.py          # NEES consistency check and Chi-squared bounds
    │       └── plotting.py       # Visualization for +/- 2-sigma and NEES
    ├── tests/                    # [NEW] Automated Testing Suite
    │   ├── conftest.py           # Shared pytest fixtures (e.g., dummy map data)
    │   ├── test_dynamics.py      # Validates F_clk and Q_clk matrices
    │   ├── test_map_manager.py   # Validates Householder transform orthogonality
    │   ├── test_measurement.py   # Validates Jacobian dimensions and chain rule
    │   └── test_stats.py         # Validates NEES Chi-squared boundary    
    ├── main.py                   # CLI entry point
    ├── pytest.ini                # Pytest configuration file
    ├── requirements.txt          # numpy, scipy, matplotlib
    └── README.md

## 3. Core Technical Specifications

### A. Dynamics & Clock Model
* **Motion**: The receiver follows a velocity random-walk model.
* **Clock**: A two-state clock model consisting of clock bias and clock drift. The clock error states are driven by white noise processes characterized by power-law coefficients h_0 and h_{-2} .

### B. Map-Assisted Constraints (Householder Transformation)
The position of the i-th virtual anchor is calculated by mirroring the true anchor position across the i-th reflecting surface.
* **Householder Matrix**: M_i = I_2x2 - 2 * n_i * n_i^T (where n_i is the unit normal vector of the wall).
* **Measurement Jacobian**: Applying the chain rule to the i-th reflected path yields: dZ_i / dr_a = -e_hat(f_i(r_a))^T * M_i .

### C. Numerical Stability & Evaluation Metrics
To prevent numerical instability and rigorously evaluate the system, the following methods are implemented:
* **Observability Rank**: Even with map assistance, if absolute time is unknown, absolute clock bias and absolute clock drift remain indistinguishable. The maximum rank is capped at 8 (instead of 10).
* **NEES Consistency Test**: The framework runs 50 Monte Carlo (MC) simulations. The average Normalized Estimation Error Squared (NEES) is plotted against theoretical Chi-squared acceptance bounds [r1, r2] to verify filter consistency .
* **Robust Matrix Inversion**: The EKF utilizes Cholesky decomposition or pseudo-inverse when computing NEES to prevent floating-point explosions in unobservable directions.

## 4. Development & Implementation Guide

### Task 1: Environment Setup
1. Implement the 2-state clock transition matrix F_clk and process noise covariance Q_clk in `dynamics.py` .
2. Create the `MapManager` to generate virtual anchor positions using the Householder transformation based on a rectangular floorplan.

### Task 2: Build the EKFs
1. **Mapless EKF**: Augment the state vector to include N virtual anchors (total dimension: 10 + 2N). Implement a trilateration warm-start to initialize the virtual anchors and prevent linearization failure.
2. **Map-assisted EKF**: Restrict the state vector to 10 dimensions. Inject the Householder matrix M_i into the anchor sub-matrix of the Jacobian .

### Task 3: Monte Carlo Runner
1. Execute 50 independent runs for each Prior Case (None, Position, Time, Pos+Time).
2. Record state estimation errors and covariance matrices at each time step.

### Task 4: Visualization
Generate three core plots:
1. **Error Trajectories**: Plot estimation error against +/- 2-sigma variance bounds.
2. **NEES Plot**: Plot the average NEES against the 99% probability region bounds (r1 and r2) .
3. **Condition Number**: Plot the singular value distribution of the Observability Matrix over time to quantify "Estimability" .

## 5. Usage

### Installation
pip install -r requirements.txt

### Running Experiments
Execute the main script by passing the target Case ID (1 through 4):
python main.py --case 1  # Runs Mapless vs Map-assisted without priors
python main.py --case 4  # Runs comparison with fully known position and time

## 6. References
1. Z. M. Kassas and T. E. Humphreys, "Observability Analysis of Collaborative Opportunistic Navigation With Pseudorange Measurements," IEEE Trans. Intell. Transp. Syst., 2014.
2. Z. Lyu and G. Zhang, "Observability Evaluation in Map-Assisted Radio SLAM." (on going)