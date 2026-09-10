# Magnetic Monopoles: Topological Solitons and Collider Phenomenology

This repository contains the numerical simulations, theoretical derivations, and source files for my research internship on magnetic monopoles, supervised by Dr. Vasiliki Mitsou at the AITANA group (IFIC-UV/CSIC). 

The project bridges formal topological gauge theories with experimental high-energy physics, specifically focusing on the production of highly-ionizing particles at colliders.

## 📖 Project Overview

The manuscript and associated code cover the following key areas:
* **Formal Gauge Theory:** Derivation of the Dirac Quantization Condition via the Aharonov-Bohm effect and the Wu-Yang construction.
* **Topological Solitons:** Analysis of the 't Hooft-Polyakov monopole, BPS limits, and dyon scattering.
* **Regularization:** Treatment of divergent energies in the electroweak Cho-Maison monopole using CKY and Born-Infeld regularization schemes.
* **Phenomenology:** Simulation of Drell-Yan monopole pair production with velocity-dependent magnetic couplings, tailored for MoEDAL detector acceptance bounds.

## 📂 Repository Structure

* `madgraph_cards/`: Contains the `.dat` configuration cards used for the MadGraph5 simulations.
* `plots/`: Output kinematic spectra generated from the ROOT analysis.
* `code/` : Contains the code used.
* `paper/`: Contains the full LaTeX source code (`.tex`, and figures) alongside the compiled PDF manuscript.

## ⚙️ Dependencies and Tools

To reproduce the simulations or compile the manuscript, the following tools are required:
* [MadGraph5_aMC@NLO](https://launchpad.net/mg5amcnlo): For event generation (Drell-Yan production).
* [ROOT (C++ / CERN)](https://root.cern/): For analyzing kinematic spectra and extracting $\beta\gamma$ distributions.
* TeX Live / pdfLaTeX: To compile the source code in the `paper/` directory.

## 🚀 Usage

### 1. Event Generation
To run the MadGraph5 simulations using the provided cards, launch `mg5_aMC` and load the configuration:
```bash
./mg5_aMC
MG5_aMC> import command madgraph_cards/run_card.dat
