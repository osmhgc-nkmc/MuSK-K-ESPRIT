[README.md](https://github.com/user-attachments/files/33952084/README.md)
# K-ESPRIT: Kinase Ensemble Sampling via Predictive Randomized Iterative Topologies

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22832131-blue)](https://doi.org/10.5281/zenodo.23657882)

This repository hosts the official implementation, datasets, and structural coordinates for:

> **Objective Identification of Basal, Primed, and Active Conformational States in Wild-Type Human MuSK Kinase via K-ESPRIT: An AlphaFold3 Ensemble and Activation-Loop Topology Mapping Framework**  
> Osamu Higuchi (2026)  

---

## Overview

The activation loop (A-loop) of muscle-specific receptor tyrosine kinase (MuSK) undergoes dynamic conformational exchange that has historically evaded static X-ray crystallography (e.g., disordered in PDB ID: `9SNU`). 

**K-ESPRIT** integrates stochastic diffusion seed perturbations in **AlphaFold3** (using full-depth, unperturbed multiple sequence alignments) with **bivariate Gaussian kernel density estimation (KDE)** along a 2D reaction coordinate:
1. **A-loop displacement:** C$\alpha$-RMSD relative to the basal ground state (residues 743–765).
2. **Hydrophobic core assembly:** Minimum heavy-atom distance between regulatory spine residues RS2 (Phe744) and RS3 (Met630).

By sampling 300 independent physical models across wild-type (WT, $N=100$), priming-mimetic (S752D, $N=100$), and constitutively active (D754A, $N=100$) constructs, K-ESPRIT objectively establishes three unmutated 3D structural medoids directly from the native WT ensemble:
* **WT Model #27 (Basal Closed):** Matches the autoinhibited crystal structure (PDB ID: `9SN0`), maintaining the persistent Asp754–Gln801 polar latch ($2.60\text{ \AA}$).
* **WT Model #45 (Primed Intermediate):** Features premature hydrophobic R-spine condensation ($d_{\min}(\text{RS2, RS3}) = 3.02\text{ \AA}$) matching the `9SNU` crystal core while retaining a compact loop.
* **WT Model #37 (Spontaneously Active):** Captures complete outward loop extension ($\text{RMSD} = 12.44\text{ \AA}$), an unbroken 4-residue R-spine, intact catalytic salt bridge (Lys609–Glu626), and active KinCore dihedrals ($\Phi = -67.1^\circ, \Psi = +100.4^\circ$), validated against the tris-phosphorylated active insulin receptor kinase benchmark (PDB ID: `1IR3`).

---

## Repository Structure

```text
MuSK-K-ESPRIT/
├── README.md                           # Project documentation and reproduction guide
├── LICENSE                             # MIT License
├── requirements.txt                    # Python environment specifications
├── .gitignore                          # Git exclusions
├── data/
│   ├── processed_metrics_with_plddt.csv # Master dataset of structural metrics and pLDDT scores
│   ├── musk_kincore_analysis.csv       # Extracted KinCore dihedrals, distances, and classes
│   ├── musk_n_lobe_plddt.csv           # P-loop and alphaC-helix pLDDT quantification
│   └── supplementary_table_1_seeds.csv # Complete seed and inference records
├── scripts/
│   ├── 01_extract_metrics.py           # Metric extraction from raw CIF ensembles
│   ├── 02_kincore_dihedrals.py         # KinCore Ramachandran & spatial distance classifier
│   ├── 03_bootstrap_convergence.py     # Monte Carlo bootstrap convergence analysis (n=1,000)
│   ├── 04_statistical_tests.py         # Mann-Whitney U, chi-square, and Fisher exact tests
│   └── 05_plot_figures.py              # Reproduction of manuscript Figures 1–3 and Supp Fig 1
├── structures/
│   └── medoids/
│       ├── WT_Model27_closed.cif       # Basal autoinhibited medoid coordinate
│       ├── WT_Model45_primed.cif       # Primed intermediate medoid coordinate
│       └── WT_Model37_active.cif       # Spontaneously active medoid coordinate
└── chimerax/
    └── visualize_medoids.cxc           # UCSF ChimeraX script for 3D overlay and annotation
