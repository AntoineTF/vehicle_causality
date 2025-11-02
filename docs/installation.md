# Installation Guide

This project relies on Python 3.9, MetaDrive, ScenarioNet, and UniTraj. The commands below reproduce the environment used for the experiments.

## 1. Create the Conda Environment

```bash
conda create -n vehicle_causality python=3.9
conda activate vehicle_causality
```

## 2. Clone the Required Repositories

```bash
cd ~/  # Choose the directory that will host the repositories
git clone https://github.com/AntoineTF/vehicle_causality.git
```

MetaDrive, ScenarioNet, and UniTraj live in sibling folders. Clone and install each project next to the main repository:

```bash
git clone https://github.com/metadriverse/metadrive.git
git clone https://github.com/AntoineTF/scenarionet.git
git clone https://github.com/AntoineTF/unitraj.git
```

## 3. Install MetaDrive

```bash
cd metadrive
pip install -e .
```

## 4. Install ScenarioNet (Modified Version)

```bash
cd ../scenarionet
pip install -e .
```

## 5. Install UniTraj (Modified Version)

```bash
cd ../unitraj
pip install -r requirements.txt
python setup.py develop
```

Return to the `vehicle_causality` repository once the dependencies are installed.
