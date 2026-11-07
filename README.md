# Stochastic HCRSP Learning

Research code for **stochastic periodic home-care routing and scheduling with caregiver learning**. The project builds caregiver schedules for recurring patient visits, models uncertain travel and service times, and evaluates the resulting schedules through simulation.

The repository contains two related experiment pipelines:

- **Short horizon**: solves independent planning instances.
- **Long horizon**: solves a sequence of instances while carrying patient acceptance, visit history, and assignment information between runs.

The optimization models include learning and no-learning variants, assignment-based variants, and deterministic-service variants. Scheduling is solved with [Gurobi](https://www.gurobi.com/), while subproblem components use IBM [CP Optimizer](https://www.ibm.com/docs/en/icos).

## Repository layout

```text
short_horizon/
  Main.py                         # Independent-instance experiments
  configuration/*.ini             # Experiment configurations
  data/real-4/*.txt               # Input instances
  result/                         # Generated analysis and summary files

long_horizon/
  Main.py                         # Sequential multi-run experiments
  configuration/*.ini             # Dynamic experiment configurations
  data/dynamic*/                  # Ordered instance sequences
  result/                         # Generated dynamic results
```

Core modules in each pipeline include:

- `Instance.py`, `Patient.py`, `Aide.py`: instance and workforce data structures
- `Instance_Reader.py`: text-instance parsing
- `Master_Problem.py`: optimization models and solution procedures
- `Subproblem.py`: routing/scheduling subproblems
- `Create_Scenario.py`: stochastic scenario generation
- `Simulation.py`: schedule evaluation under simulated uncertainty

## Requirements

- Python 3.9 or newer
- A licensed and configured Gurobi installation
- IBM ILOG CP Optimizer and the `docplex` Python package
- Python packages:

  ```bash
  python -m pip install numpy pandas scipy gurobipy docplex
  ```

Both solvers must be available to Python before running an experiment. Solver licensing and installation are external prerequisites; this repository does not include solver binaries or licenses.

## Running experiments

Run commands from the directory containing the selected `Main.py`, because the scripts use local module imports:

```bash
cd short_horizon
python Main.py
```

For sequential dynamic experiments:

```bash
cd long_horizon
python Main.py
```

`Main.py` reads every `.ini` file in the corresponding `configuration/` directory. To run one configuration at a time, temporarily leave only the desired configuration file in that directory or move other configurations elsewhere.

The default configurations demonstrate:

- `short_horizon/configuration/config-learning.ini`: learning with stochastic service times
- `short_horizon/configuration/config-no-learning.ini`: no-learning comparison
- `long_horizon/configuration/config-learning.ini`: sequential learning experiment
- `long_horizon/configuration/config-no-learning.ini`: sequential no-learning comparison

Edit the `[PARAMETERS]` section to select the instance case, learning mode, assignment mode, team size, scenario count, service-time assumptions, and other model parameters. In the long-horizon configurations, `scenario_instance` defines the instance used to create the shared simulation scenario and `instance_files` defines the ordered run sequence.

## Input format and data

Input instances are plain-text files stored under each pipeline's `data/` directory. Existing files cover different patient and caregiver counts, for example:

```text
short_horizon/data/real-4/instance-patient-25-aide-5.txt
long_horizon/data/dynamic/instance-patient-30-aide-5.txt
```

The included data and configuration files are intended for reproducing the repository's experiments. If you add a new instance, place it in the case directory referenced by `case` and ensure that its filename is included by the selected configuration.

## Outputs

Runs write logs and generated analysis files under the relevant `result/` directory. The short-horizon pipeline also includes `short_horizon/result/data_summary.csv`, which records instance size, accepted visits, service-level statistics, routing statistics, and solving time.

Long-horizon runs additionally produce continuity-of-care results showing how assignments and visit history evolve across the configured sequence of instances.

## Reproducibility notes

- Configuration files are the source of truth for model and simulation parameters.
- Scenario generation and simulation can use thousands of scenarios (`simulation_scenario` is `10000` in the example configurations).
- Optimization and simulation may take substantial time as the number of patients, visits, caregivers, or scenarios increases.
- Generated result files should be treated as experiment artifacts and are not required to run the models.

## Citation

If you use this repository in academic work, please cite the associated publication or contact the repository authors for the preferred citation.
