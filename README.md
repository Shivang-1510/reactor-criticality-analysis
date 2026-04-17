# Reactor Criticality Analysis using Neutron Diffusion

## Overview

Criticality in a nuclear reactor is the stable state where a sustained, controlled fission chain reaction occurs, producing a constant power output. It signifies a perfect balance between neutron production and neutron loss within the reactor core. This project models neutron behavior in a nuclear reactor using the neutron diffusion equation and computes the effective multiplication factor (k-effective). It determines whether the reactor is subcritical, critical, or supercritical. Subcritical: Chain reaction dies out (neutron loss > production).
Critical: Chain reaction is stable (production = loss).
Supercritical: Chain reaction increases (production > loss), used to raise power levels.

## Objectives

* Solve neutron diffusion equation numerically
* Compute k-effective using power iteration
* Analyze reactor stability conditions
* Visualize neutron flux distribution

## Theory

The governing equation:

-D d²φ/dx² + Σaφ = (1/k) νΣfφ

Where:

* φ = neutron flux
* Σa = absorption cross-section
* νΣf = neutron production term
* k = multiplication factor

## Methodology

* Discretized reactor into a 1D grid
* Applied finite difference method
* Used power iteration to compute k-effective
* Normalized neutron flux for numerical stability

## Results

The simulation outputs:

* k-effective value
* Reactor state (subcritical, critical, supercritical)
* Neutron flux distribution across reactor

## Visualization



## Key Insights

* Reactor criticality depends on neutron production vs absorption
* Flux peaks at the reactor center
* Small parameter changes significantly affect k-effective

## Tools Used

* Python
* NumPy
* Matplotlib

## How to Run

```bash
pip install -r requirements.txt
python src/main.py
```

## Applications

* Reactor physics analysis
* Nuclear reactor design fundamentals
* Safety and stability evaluation

## Future Work

* Extend to 2D reactor core
* Add control rod reactivity effects
* Include time-dependent neutron behavior

## Author

Shivang Arora
Energy Engineering Student | Reactor Physics Enthusiast
