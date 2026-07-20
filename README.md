# Counterfactual Explanations for Optimization Problems

This repository contains implementations and experiments for **counterfactual
explanations (CE)** in optimization problems — with a focus on *weak* and
*relative* counterfactuals and their application to power-system unit-commitment
models.

## Project Structure

```
.
├── notebooks/                 # Method notebooks, grouped by problem class
│   ├── linear/                #   LP-based CE (relative CE, weak CE, WCEP knapsack, RCEP analysis)
│   ├── integer/               #   Integer-domain weak CE (dimension-3 study)
│   ├── basbl/                 #   BASBL prototype
│   ├── mixed/                 #   NCXplain, weak CE for mixed problems, parallelized algorithm test
│   └── wcep_leftraru/         #   WCEP decarbonization planning experiments (SLURM / Leftraru)
│
├── uc_experiments/            # Unit-Commitment CE engine and experiments
│   ├── uc_pipeline.py         #   Network-UC model builders and solver
│   ├── uc_data_loader.py      #   Test-system loaders (quick_setup)
│   ├── uc_decomp_4b.py        #   DECOMP solver
│   ├── uc_branch_sandwich_4b.py, uc_master_relax_4b.py
│   ├── Data/                  #   IEEE test systems (14 / 30 / 39 / 57 / 118 / 300 bus)
│   ├── dev/                   #   Diagnostic and reproduction scripts
│   ├── pipeline_results/, bs_results/, benchmark_results/   # committed results
│   ├── DECOMP_*.md            #   Solver design / state notes (also used as run-time offload state)
│   └── *.slurm                #   Leftraru batch scripts
│
├── requirements.txt
└── README.md
```

Large experiment output dumps (SLURM logs, per-box JSON, `*_a05_*` campaigns) are
regenerated on the cluster and are intentionally **git-ignored** — see `.gitignore`.

## Dependencies

Install the required packages using:

```bash
pip install -r requirements.txt
```

### Gurobi License

This project uses the [Gurobi](https://www.gurobi.com/) optimizer, which requires a
valid license. Set the `GRB_LICENSE_FILE` environment variable to point to your
license file, e.g.:

```bash
export GRB_LICENSE_FILE=/path/to/gurobi.lic     # Linux / macOS
$env:GRB_LICENSE_FILE = "C:\path\to\gurobi.lic" # Windows PowerShell
```

## Usage

The notebooks under `notebooks/` are self-contained; open them from their own
directory so that relative data paths resolve.

The unit-commitment experiments run from `uc_experiments/` — the modules import one
another by name, so run scripts and notebooks from inside that directory:

```bash
cd uc_experiments
python run_pipeline.py            # local run
sbatch run_pipeline.slurm         # on the Leftraru (NLHPC) cluster
```

Data paths are resolved relative to each script/notebook, so no absolute paths need
editing after cloning.

## Contributing

This is a research repository accompanying a paper in preparation. Please open an
issue before submitting substantial changes.
