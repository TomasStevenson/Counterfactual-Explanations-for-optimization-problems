# Security-Constrained Decarbonization Planning

This folder contains the implementation and data used for the
security-constrained decarbonization planning case study (Query III).

The model considers multi-year generation dispatch, investment in
inverter-based resources (IBRs), and frequency-security requirements.

## Main file

- `WCEP_Descarbonizacion_Unificado.ipynb`  
  Main notebook containing the forward planning model and the
  counterfactual explanation formulations.

## Test systems

The experiments are performed on the IEEE 14-, 39-, and 57-bus systems.

The folder includes the processed generation, demand, renewable-profile,
and network data required to reproduce the experiments.

## Counterfactual parameters

Two families of mutable parameters are considered:

- Maximum capacities of synchronous generators (`Pmax`)
- Inertia requirements and IBR investment bounds

The desired set imposes a reduction in emissions relative to the
forward solution.

## Solution methods

The counterfactual problem is reformulated using primal feasibility,
dual feasibility, and strong duality.

Two solution strategies are included:

1. Direct solution of the resulting nonconvex bilinear formulation
   using Gurobi.
2. PADM-W used as a warm start, followed by the same bilinear
   strong-duality formulation.

## Requirements

A valid Gurobi license is required.

The main Python dependencies are listed in the repository-level
`requirements.txt`.
