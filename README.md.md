# ScenarioNet: Key Components and Functions in the Pipeline

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

---

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

## 📊 Example Use Cases for This Pipeline

- **Causal Analysis:** Study how removing vehicles affects collision probabilities.
- **Trajectory Prediction Evaluation:** Generate complex driving situations for testing trajectory prediction models.
- **Reinforcement Learning Training:** Create diverse driving datasets for policy learning in self-driving cars.