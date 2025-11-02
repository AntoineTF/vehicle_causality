# Dataset Overview

This project combines large-scale synthetic data with the nuScenes dataset and several out-of-distribution (OOD) benchmarks. All datasets follow the UniTraj format.

## Synthetic Dataset

- Hosted on EPFL SCITAS: `/work/vita/datasets/vehicle_causality/merged_cf_f`
- 160,000 total scenarios:
  - 110,000 for training
  - 40,000 for testing

## nuScenes Dataset

- Hosted on EPFL SCITAS: `/work/vita/datasets/Scenarionet_Dataset`
- Folder layout:
  - `train/nuscenes`
  - `validation/nuscenes`

## Out-of-Distribution Benchmarks

Two synthetic variants probe generalization capabilities:

1. **Modified Traffic Density**  
   `/work/vita/datasets/vehicle_causality/ood/large_datasets/sim_ood_td_merged`
2. **Alternative Control Policy (IDM)**  
   `/work/vita/datasets/vehicle_causality/ood/large_datasets/sim_ood_idm_merged`

Each OOD dataset contains roughly 2,000 simulations.

## UniTraj Data Fields

- `scenario_id`: Unique identifier for each driving situation.
- `obj_trajs`: Historical trajectories with position, size, type, heading, velocity, and acceleration.
- `obj_trajs_mask`: Valid mask for available data points.
- `map_center`: Map origin point.
- `map_polylines`: Lane and boundary polylines.
- `obj_trajs_future_state`: Future trajectory predictions.
- `obj_trajs_future_mask`: Valid mask for future predictions.
- `center_gt_trajs`: Ground-truth trajectory for the ego agent.

## Output Directory Structure

```
simulation/
├── dataset_mapping.pkl
├── dataset_summary.pkl
├── simulation_0/
│   ├── dataset_mapping.pkl
│   ├── dataset_summary.pkl
│   ├── sd_pg_MetaDrive_v0.4.2.3_PGMap-0_start_78.pkl
│   ├── sd_pg_MetaDrive_v0.4.2.3_PGMap-0_start_78_child_0.pkl
│   └── ...
└── simulation_1/
    └── (same layout as above)
```

Counterfactual variations appear as `*_child_*.pkl` files paired with the main factual scenario.
