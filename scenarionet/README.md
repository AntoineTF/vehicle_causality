# ScenarioNet: Key Components and Functions in the Pipeline

## How to Run the Pipeline
To run the entire scenario generation pipeline, use the following command in your terminal:
```bash
python convert_pg.py --database_path /path/to/save/data --num_scenarios 100 --num_workers 8 --start_index 0 --overwrite
```
### Key Arguments Explained:
- `-d`or `--database_pat`: Path wehere the generated data will be stored.
- `--num_scenarios`:Number of scenarios to generate.
- `--num_workers`: Number of CPU cores for parallel processing.
- `--start_index`: Starting index for scenario generation (useful for parallelization).
- `--overwrite`: If set, existing data in the specified path will be overwritten.

## Key Components and Functions in the Pipeline

### **1. `convert_pg.py` (Main File)**

- **Role:** Entry point for running the entire generation process.
- **Key Functions Called:**
   - `get_pg_scenarios`: Generates a list of scenario indices to process.
   - `write_to_directory`: Distributes tasks for parallel scenario generation.

---

### **2. `write_to_directory` (Parallel Processing Management)**

- **Role:** Manages the parallelization of scenario generation.
- **Process:**
   - Splits the scenario indices among multiple workers.
   - Launches parallel workers using `multiprocessing.Pool`.
   - Each worker calls `write_to_directory_single_worker`.

---

### **3. `write_to_directory_single_worker` (Core Generation Logic)**

- **Role:** Manages the generation of a batch of scenarios in each worker.
- **Process:**
   - Initializes the simulation environment.
   - Calls `convert_pg_scenario` for each scenario.
   - Filters out invalid scenarios (e.g., scenarios where vehicles crashed).
   - Saves the valid scenarios and counterfactual variations as pickle files.

---

### **4. `convert_pg_scenario` (Scenario Simulation)**

- **Role:** Simulates a driving scenario and generates counterfactual variations.
- **Key Features:**
   - Simulates driving using a default policy.
   - **Crash Filtering:** Scenarios where a crash occurred are automatically removed.
   - **Complexity Detection:** Identifies the most complex moments using a Kalman filter (`find_most_complex_timeframe`).
   - **Counterfactual Generation:** Removes selected vehicles and reruns the scenario to observe differences.

---

### **5. `find_most_complex_timeframe` (Complexity Analysis)**

- **Role:** Identifies critical driving moments based on prediction error.
- **Process:**
   - **Kalman Filtering:** Predicts future positions of the ego vehicle using a Kalman filter.
   - **Error Calculation:**  Compares predicted vs. actual future positions.
   - **Critical Moment Detection:** Time steps with high prediction errors are selected for counterfactual generation.

---

### **6. `update_scenario_for_timestep` (Scenario Cropping)**

- **Role:** Crops the scenario to focus on a critical timeframe identified in `find_most_complex_timeframe`.
- **Process:**
   - Adjusts the metadata and time length for the cropped section.
   - Ensures data consistency with padding if needed.

---

### **7. `calculate_continuous_valid_length` (Data Quality Check)**

- **Role:** Validates the scenario data for continuous valid entries.
- **Process:**
   - Checks how long a vehicle remained in a valid state (not crashing or going off-road).

## 📁 Output File Structure
When you run the pipeline and specify a directory (e.g., `--database_path simulation`), the pipeline will create a structured dataset for procedural driving simulations and counterfactual analysis.

### 📦 Folder Structure Overview:
```php
simulation/                                                     # Main output folder
│
├── dataset_mapping.pkl                                         # Summary mapping for all simulations in the dataset
├── dataset_summary.pkl                                         # Overview and metadata for all simulations
│
├── simulation_0/                                               # Subfolder for the first worker's batch of scenarios
│   ├── dataset_mapping.pkl                                     # Summary mapping for this worker's simulations
│   ├── dataset_summary.pkl                                     # Metadata summary for this worker's scenarios
│   ├── sd_pg_MetaDrive v0.4.2.3_PGMap-0_start_78.pkl           # Main scenario file
│   ├── sd_pg_MetaDrive v0.4.2.3_PGMap-0_start_78_child_0.pkl   # Counterfactual file 1
│   ├── sd_pg_MetaDrive v0.4.2.3_PGMap-0_start_78_child_1.pkl   # Counterfactual file 2
│   └── ...                   
│
├── simulation_1/                                               # Subfolder for the second worker's batch of scenarios
│   └── (Same structure as above)
│
└── ...                                                         # More simulation subfolders depending on the number of workers

```
#### 📊 **Main Files:**
- `dataset_mapping.pkl` (in the root and subfolders):
    - Contains a mapping of all generated scenario files.
    - Provides easy access to locate each scenario and its counterfactuals.
- `dataset_summary.pkl` (in the root and subfolders):
    - Provides a summary of the entire dataset including metadata such as the number of scenarios, number of counterfactuals generated, and scenario lengths.
#### 📈 **Simulation Files:**
##### **1. Primary Simulation File:**
- Example: `sd_pg_MetaDrive v0.4.2.3_PGMap-0_start_78.pkl`
- Expanation: 
    - `"sd_pg_MetaDrive v0.4.2.3_PGMap"`: This prefix is standard for all procedural generation scenarios.
    - `-0`: Refers to the map ID. Here, the scenario was generated on map 0.
    - `start_78`: This indicates that the simulation started from timestep 78 instead of the beginning. The starting timestep was chosen because the algorithm identified a complex driving interaction at this point, determined using the `find_most_complex_timeframe` function.
##### **2. Counterfactual Files:**
- Example: `sd_pg_MetaDrive v0.4.2.3_PGMap-0_start_78_child_0.pkl`
- Expanation: 
    - `"child_0"`: indicates the first counterfactual scenario generated from the original simulation.
    - Each counterfactual file represents a modified version of the primary simulation where one agent (vehicle) was removed to study the causal effect of its presence.
    - There will be as many child files as there are non-ego vehicles in the original scenario.

## 🔧 Post-Processing with `merging_c_cf.py`
After running the ScenarioNet pipeline, you need to merge the factual and counterfactual files into a single file to prepare the data for use in UniTraj.
Run the following command:
```python
python merging_c_cf.py --base_directory /path/to/simulation --output_directory /path/to/merged_data
```
### What it Does:
- Combines the factual and counterfactual files for each simulation.
- Formats the dataset correctly for use with UniTraj.
- Merges metadata for easier downstream use.

## 🛠️ Customizing Scenario Generation

You can customize various aspects of the simulation by adjusting parameters in the **`convert_pg.py`** file:

**Modifiable Parameters:**
- **Traffic Density (`traffic_density`)**: Adjust how many vehicles populate the map.
- **Policy (`custom_policy`)**: Choose between `IDMPolicy` (Intelligent Driver Model) or `ExpertPolicy`.
- **Accident Probability (`accident_prob`)**: Set the probability of a crash occurring in the scenario.
- **Custom Seeds (`custom_seeds`)**: Use fixed seeds for reproducibility.

## 🎯 Key Insights and Features

- ✅ **Crash Filtering:** Scenarios where a crash occurs are automatically discarded to maintain dataset quality.
- ✅ **Counterfactual Generation:** Each complex moment can be modified by selectively removing vehicles and rerunning the simulation.
- ✅ **Parallelization:** The process can run across multiple CPU cores for faster data generation.
- ✅ **Reproducibility:** Custom seeds and fixed vehicle configurations ensure reproducible results.
- ✅ **Flexible Customization:** Traffic density, policies, and other settings can be adjusted for experimentation.

## 📖 Citation

This work builds on **ScenarioNet**, an open-source platform for large-scale traffic scenario simulation and modeling. If you use this repository or extend it for your own work, please cite the original authors:

```bibtex
@article{li2023scenarionet,
  title={ScenarioNet: Open-Source Platform for Large-Scale Traffic Scenario Simulation and Modeling},
  author={Li, Quanyi and Peng, Zhenghao and Feng, Lan and Liu, Zhizheng and Duan, Chenda and Mo, Wenjie and Zhou, Bolei},
  journal={Advances in Neural Information Processing Systems},
  year={2023}
}
```