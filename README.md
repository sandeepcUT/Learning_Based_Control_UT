# Learning Based Optimal Control - UT Austin ECE 381V

This repository contains code examples and implementations for the UT Austin ECE 381V course on Learning Based Optimal Control.

## Project Structure

```
Learning_Based_Control_UT/
├── README.md                                    # This file
├── main.py                                      # Main entry point
├── pyproject.toml                               # Project dependencies and configuration
├── uv.lock                                      # Lock file for reproducible builds
├── motivation_optimal_control_Lecture_2/        # Lecture 2 examples
│   ├── lqr_control.py                          # LQR control with control constraints
│   └── PID_control.py                          # PID control parameter analysis
└── Lecture3_Electricity_Grid_DP/               # Lecture 3 examples
    └── load_scheduling.py                      # Load scheduling with slew rate constraints
```

## Files Description

### Lecture 2 - Motivation for Optimal Control

- **`lqr_control.py`**: Demonstrates Linear Quadratic Regulator (LQR) control with different control constraints. Shows how varying the maximum control norm affects the optimal trajectory and control effort.

- **`PID_control.py`**: Analyzes PID controller performance by varying proportional (kp) and derivative (kd) gains. Visualizes the effect of different parameter values on system response.

### Lecture 3 - Dynamic Programming Applications

- **`load_scheduling.py`**: Implements load scheduling optimization for electricity generation with slew rate constraints. Compares different constraint levels and their impact on generation costs.

## Prerequisites

- Python 3.13 or higher
- [uv](https://docs.astral.sh/uv/) package manager

## Installation

### 1. Install uv

**On macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**On Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Alternative installation methods:**
- Using pip: `pip install uv`
- Using homebrew (macOS): `brew install uv`
- Using cargo: `cargo install uv`

For more installation options, visit: https://docs.astral.sh/uv/getting-started/installation/

### 2. Initialize the project

```bash
# Clone the repository (if not already done)
git clone <repository-url>
cd Learning_Based_Control_UT

# Install dependencies using uv
uv sync

# Activate the virtual environment
uv shell
```

## Running the Code

### Individual Examples

**LQR Control (Lecture 2):**
```bash
python motivation_optimal_control_Lecture_2/lqr_control.py
```

**PID Control Analysis (Lecture 2):**
```bash
python motivation_optimal_control_Lecture_2/PID_control.py
```

**Load Scheduling (Lecture 3):**
```bash
python Lecture3_Electricity_Grid_DP/load_scheduling.py
```

**Main entry point:**
```bash
python main.py
```

### Using uv to run scripts

You can also run scripts directly with uv:

```bash
# Run individual examples
uv run motivation_optimal_control_Lecture_2/lqr_control.py
uv run motivation_optimal_control_Lecture_2/PID_control.py
uv run Lecture3_Electricity_Grid_DP/load_scheduling.py

# Run main script
uv run main.py
```

## Dependencies

The project uses the following Python packages (see `pyproject.toml`):

- **cvxpy** (≥1.7.2): Convex optimization library
- **matplotlib** (≥3.10.6): Plotting and visualization
- **numpy** (≥2.3.3): Numerical computing
- **scipy** (≥1.16.2): Scientific computing

## Course Information

This code accompanies the UT Austin ECE 381V course on Learning Based Optimal Control, covering:

- Optimal control theory fundamentals
- Linear Quadratic Regulator (LQR) design
- PID controller analysis
- Dynamic programming applications
- Convex optimization in control systems

## Contributing

This is a course repository. For questions or issues, please contact the course instructor.

## License

This project is for educational purposes as part of the UT Austin ECE 381V course.
