# Argument-based opinion dynamics with confirmation bias: a model translation exercise.

This repository contains code and data for the paper:

**"Accurate mean-field predictions for cognitively grounded social influence dynamics with confirmation bias"**

**Abstract:** Collective opinion dynamics in human populations are frequently studied using cognitively detailed agent-based models, but their high dimensionality often limits analytical insight. Here, we demonstrate that a class of argument-exchange models incorporating confirmation bias can be reduced to a low-dimensional and analytically tractable mean-field description. Our approach maps pairwise interaction rules onto an influence–response function, which is subsequently aggregated into a two-compartment dynamical system whose Jacobian separates into a mean mode and a polarization mode. This yields a simple stability test that diagnoses the onset of symmetry breaking. The framework captures the transition from consensus to polarization and reveals a second critical threshold beyond which polarized states become fully stable. Across parameter regimes, the reduced system closely reproduces the dynamics observed in the underlying agent-based model. These findings provide a general, mechanism-based perspective on how cognitive biases influence collective outcomes and show that complex social influence processes can exhibit simple and tractable phase-space structure.

## Overview

We study a cognitively grounded model of opinion dynamics based on argument exchange with confirmation bias. Analyses have been performed in MatLab, Mathematica and Python. 

### `matlab/` includes:

- The original agent-based model (PAT), following [Banisch and Shamon (2025)](https://doi.org/10.1177/00491241231186658)
- The reduced model based on a derived influence–response function (IRF)
- Several mean-field (MF) models and analysis of the system
- Scripts to reproduce the main figures in the paper

### `mathematica/` includes:

- The derivation of the influence response function
- Mean-field model and bifurcation analysis

### `python/` includes:

- The original agent-based model (PAT), following [Banisch and Shamon (2025)](https://doi.org/10.1177/00491241231186658)
- The reduced model based on a derived influence–response function (IRF)
- Notebooks for their comparison (in the SI of the paper)

## Contact

For questions, please contact the corresponding author.