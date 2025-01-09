# Vehicle Causality Project: Causal Analysis for Autonomous Driving

This project introduces a framework for **causal analysis** in autonomous driving simulations, designed to investigate how the presence or absence of specific vehicles influences the behavior of the ego vehicle.

### Key Contributions:

- **Causal Analysis Framework:** The framework allows selective removal of agents from driving scenarios to assess their causal impact on the ego vehicle.
- **Causal Regularization:** We introduce two types of causal regularization techniques **contrastive loss** and **ranking loss**  both aimed at enhancing causal awareness and improving model generalization to **out-of-distribution (OOD)** scenarios.
- **Counterfactual Generation:** Procedural scenarios and counterfactual variations are automatically generated, enabling a structured approach to assess the impact of individual agents.


## Installation

Follow these steps to set up the environment and required tools:

```bash
# Create a dedicated Conda environment
conda create -n vehicle_causality python=3.9
conda activate vehicle_causality

# Clone the repository
cd ~/  # Go to the folder you want to host these three repos.
git clone https://github.com/AntoineTF/vehicle_causality.git

# Install MetaDrive Simulator
cd metadrive
pip install -e .

# Install the modified version of ScenarioNet
cd ..
cd scenarionet
pip install -e .

# Install the modified version of UniTraj
cd ..
cd unitraj 
pip install -r requirements.txt
python setup.py develop
```

## Dataset information

### Synthetic Dataset:

The synthetic dataset used in this project is available on **EPFL SCITAS** at:

`/work/vita/datasets/vehicle_causality/merged_cf_f`

- **160,000 total scenarios:**
    - **110,000 for training**
    - **40,000 for testing**

### NuScenes Dataset:

The **nuScenes** dataset is available at:

`/work/vita/datasets/Scenarionet_Dataset`

- The structure includes a **train** and **validation** folder, each containing a `nuscenes` subfolder.

## Out-of-Distribution (OOD) Datasets

Two **out-of-distribution (OOD)** datasets were created for evaluating model generalization:

1. **Modified Traffic Density:** Adjusted vehicle density to test performance under varying traffic conditions.
2. **Different Policy Control:** Altered the control policy for non-ego agents.

Each dataset contains approximately **200 simulations**.

## Baseline Experiments

We provide three baseline experiments to evaluate performance:

1. **AutoBot on the NuScenes dataset.**
2. **AutoBot on the synthetic dataset.**
3. **AutoBot on a combined dataset (NuScenes + Synthetic).**


### Configuration Setup for Training

Before running the experiments, you need to modify the `config.yml` file located in `unitraj/config/` by adjusting the following fields:

### **For NuScenes Dataset:**

```yaml
train_data_path: ["/work/vita/datasets/Scenarionet_Dataset/train/nuscenes"]
val_data_path: ["/work/vita/datasets/Scenarionet_Dataset/validation/nuscenes"]
max_data_num: [null]
starting_frame: [0]

```

### **For Synthetic Dataset:**

```yaml
train_data_path: [
    "/work/vita/datasets/vehicle_causality/merged_cf_f/train/simulation1",
    "/work/vita/datasets/vehicle_causality/merged_cf_f/train/simulation18",
    ...
]
val_data_path: [
    "/work/vita/datasets/vehicle_causality/merged_cf_f/validation/simulation36",
    "/work/vita/datasets/vehicle_causality/merged_cf_f/validation/simulation37",
    ...
]
max_data_num: [null, null,...]
starting_frame: [0, 0,...]
```

## Running the Experiments

### **Training a Model with Causal Regularization:**

To train a model with causal regularization, update the **`autobot.yaml`** file:

- Set **`ret_embeddings: True`** (enables causal metrics).
- Adjust the regularization type (`reg_type`):
    - `"contrastive"`: for contrastive causal regularization
    - `"ranking"`: for ranking-based regularization
    - `"no_reg"`: for no causal regularization

Run the training:

```bash
python train.py method=autobot
```

---

**Evaluation:**

To evaluate the model on a validation set:

1. Set the `ckpt_path` in `config.yaml` to point to your trained model checkpoint.
2. Ensure the `val_data_path` points to the correct validation set.

Run the evaluation script:

```bash
python evaluation.py
```

## Dataset Structure (for UniTraj)

The datasets follow the **UniTraj** structure. Here's a brief summary of the key components:

- **Scenario Metadata:**
    - `scenario_id`: Unique identifier for each scenario.
- **Object Trajectories:**
    - `obj_trajs`: Historical trajectories with information about position, size, type, heading, velocity, and acceleration.
    - `obj_trajs_mask`: Valid mask indicating available data points.
- **Map Information:**
    - `map_center`: Center point of the map.
    - `map_polylines`: Polyline representations of lanes and boundaries.
- **Future State Predictions:**
    - `obj_trajs_future_state`: Future trajectory predictions.
    - `obj_trajs_future_mask`: Mask for valid predictions.
- **Ground Truth Data:**
    - `center_gt_trajs`: Ground truth for the centered agent.

## Merging Counterfactual Files

After running **ScenarioNet (see ScenarioNet folder)**, the factual and counterfactual simulations need to be merged for compatibility with **UniTraj**. Run the following command:

```bash
python merging_c_cf.py --base_directory <path_to_data> --output_directory <new_path>
```

## 📂 Output File Structure

The generated datasets are structured as follows:

```
plaintext
Copier le code
simulation/                              # Main output folder
├── dataset_mapping.pkl                  # Summary mapping for all simulations
├── dataset_summary.pkl                  # Overview of the dataset
├── simulation_0/                        # Batch of scenarios for worker 0
│   ├── dataset_mapping.pkl
│   ├── dataset_summary.pkl
│   ├── sd_pg_MetaDrive_v0.4.2.3_PGMap-0_start_78.pkl   # Main scenario
│   ├── sd_pg_MetaDrive_v0.4.2.3_PGMap-0_start_78_child_0.pkl  # Counterfactual
│   ├── sd_pg_MetaDrive_v0.4.2.3_PGMap-0_start_78_child_1.pkl
│   └── ...
└── simulation_1/                        # Batch of scenarios for worker 1
    └── (Same structure as above)

```

## Customization Options

You can adjust the following parameters directly in the `convert_pg.py` and `autobot.yaml` files:

- **Traffic Density:** Adjust the number of vehicles per scenario.
- **Policy Selection:** Choose between `IDMPolicy` or `ExpertPolicy`.
- **Crash Probability:** Modify the likelihood of crashes in scenarios.
- **Random Seeds:** Ensure reproducibility using fixed random seeds.

## Citation

If you use this repository in your research, please cite **ScenarioNet**, **MetaDrive** and **UniTraj**:

```
@article{li2023scenarionet,
  title={ScenarioNet: Open-Source Platform for Large-Scale Traffic Scenario Simulation and Modeling},
  author={Li, Quanyi et al.},
  journal={Advances in Neural Information Processing Systems},
  year={2023}
}
@article{li2022metadrive,
  title={MetaDrive: Composing Diverse Driving Scenarios for RL},
  author={Li, Quanyi et al.},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2022}
}
@article{feng2024unitraj,
  title={UniTraj: A Unified Framework for Scalable Vehicle Trajectory Prediction},
  author={Feng, Lan and Bahari, Mohammadhossein and Amor, Kaouther Messaoud Ben and Zablocki, {\'E}loi and Cord, Matthieu and Alahi, Alexandre},
  journal={arXiv preprint arXiv:2403.15098},
  year={2024}
}
```