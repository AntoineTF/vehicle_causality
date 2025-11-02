# Training and Evaluation

This guide covers experiment configuration, training commands, evaluation, and data processing utilities.

## Baseline Experiments

Three reference configurations are provided:

1. AutoBot trained on nuScenes.
2. AutoBot trained on the synthetic dataset.
3. AutoBot trained on the merged nuScenes + synthetic data.

## Configure UniTraj

Update `unitraj/config/config.yml` before launching experiments.

### nuScenes

```yaml
train_data_path: ["/work/vita/datasets/Scenarionet_Dataset/train/nuscenes"]
val_data_path: ["/work/vita/datasets/Scenarionet_Dataset/validation/nuscenes"]
max_data_num: [null]
starting_frame: [0]
```

### Synthetic

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
max_data_num: [null, null, ...]
starting_frame: [0, 0, ...]
```

## Train with Causal Regularization

Edit `unitraj/config/autobot.yaml`:

- `ret_embeddings: True` (also enable in `config.yaml`).
- `reg_type`: choose `"contrastive"`, `"ranking"`, or `"no_reg"`.

Launch training:

```bash
python train.py method=autobot
```

## Evaluate

1. Set `ckpt_path` in `config.yaml` to your checkpoint.
2. Confirm `val_data_path` matches the target validation split.

```bash
python evaluation.py
```

## Merge Factual and Counterfactual Files

Use `merging_c_cf.py` to align ScenarioNet outputs with the UniTraj format:

```bash
python merging_c_cf.py --base_directory <path_to_data> --output_directory <new_path>
```

## Customisation Options

Tune the data generation scripts (`convert_pg.py`) and model config (`autobot.yaml`) to explore new conditions:

- Traffic density.
- Control policy (`IDMPolicy` vs `ExpertPolicy`).
- Crash probability.
- Random seeds for reproducibility.
