import logging
import pickle
import os
import random 
import numpy as np
import time

from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.policy.expert_policy import ExpertPolicy
from metadrive.scenario.scenario_description import ScenarioDescription as SD


def make_env(start_index, num_scenarios, extra_config=None):
    config = dict(
        start_seed=start_index,
        num_scenarios=num_scenarios,
        traffic_density=random.uniform(0.17,0.23),
        agent_policy=IDMPolicy,
        accident_prob=0,
        crash_vehicle_done=False,
        crash_object_done=False,
        store_map=False,
        map=1
    )
    extra_config = extra_config or {}
    config.update(extra_config)
    env = MetaDriveEnv(config)
    return env


def convert_pg_scenario(scenario_index, version, env):
    """
    Simulate and export a Procedurally Generated (PG) scenario from MetaDrive.

    This function simulates a driving scenario, collects the trajectory data, and processes it 
    based on crash status and trajectory length. If the scenario is invalid, it gets skipped.

    Parameters:
    - scenario_index (int): The index of the scenario to export (within a specific batch).
    - version (str): The dataset version identifier for validation.
    - env (MetaDriveEnv): The MetaDrive environment instance used to generate the scenario.

    Returns:
    - dict or None: Returns the processed scenario if valid, otherwise None.
    """

    # Start timing the process for performance monitoring
    start_time = time.time()
    print("\n" + "="*50)
    print(f"Starting scenario {scenario_index}...")
    
    # Configuration for scenario validation
    len_short = False
    length_threshold = 60           # Minimum length required for a valid scenario
    logging.disable(logging.INFO)   # Suppress excessive logging for clarity
    policy = lambda x: [0, 1]       # Placeholder driving policy (constant acceleration)
    
    # Export the scenario using MetaDrive's internal method
    # - The `policy` dictates the agent's behavior
    # - The scenario will run for a maximum of 500 steps
    # - The output is not converted to a dictionary yet
    scenarios, done_info = env.export_scenarios(
        policy, 
        scenario_index=[scenario_index],
        max_episode_length=500, 
        suppress_warning=True, 
        to_dict=False
    )
    
    # Extract the scenario and relevant information
    scenario = scenarios[scenario_index]
    scenario_length = scenario["length"] 
    ego_id = scenario["metadata"]["sdc_id"]  # SDC = Self-Driving Car ID
    positions = scenario["tracks"][ego_id]["state"]["position"]
    
    # Check if the scenario ended due to a crash
    print("done_info[crash]: ", done_info[scenario_index].get("crash", "unknown")) 
    if done_info[scenario_index].get("crash", "unknown"):
        print(f"Skipping scenario {scenario_index} due to a crash.")
        return None
    
    # Handle scenarios shorter than the defined length threshold
    if scenario_length < length_threshold:
        print(f"Skipping scenario {scenario_index} due to insufficient length.")
        history_len = scenario_length // 3
        prediction_horizon = scenario_length - history_len
        skip_initial_steps = 0
        # Attempt to find a more complex portion of the scenario for better coverage
        best_scenario_index,_ = find_most_complex_timeframe(
            positions, 
            lower_threshold=100, 
            history_len=history_len, 
            prediction_horizon=prediction_horizon, 
            skip_initial_steps=skip_initial_steps
        )
        # Cropping is commented out; original scenario returned instead
        # scenario_cropped = update_scenario_for_timestep(scenario, best_scenario_index, length=length_threshold)
        scenario_cropped = scenario
        
    # If the scenario length is sufficient, find a complex portion for better testing
    else :
        best_scenario_index,_ = find_most_complex_timeframe(positions, lower_threshold=100)
        # Cropping is commented out; original scenario returned instead
        # scenario_cropped = update_scenario_for_timestep(scenario, best_scenario_index, length=length_threshold)
        scenario_cropped = scenario

    # Ensure version consistency across datasets
    assert scenario_cropped[SD.VERSION] == version, "Data version mismatch"
    
    # End timing the process and log duration
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("="*50 + "\n")          
    
    # Return the cropped scenario data for further processing or saving
    return scenario_cropped


def get_pg_scenarios(start_index, num_scenarios):
    return [i for i in range(start_index, start_index + num_scenarios)]

def estimate_kalman_filter(history, prediction_horizon):
    """
    Predict the future position by running the kalman filter.

    :param history: 2d array of shape (length_of_history, 2)
    :param prediction_horizon: how many steps in the future to predict
    :return: the predicted position (x, y)

    Code taken from:
    On Exposing the Challenging Long Tail in Future Prediction of Traffic Actors
    """
    length_history = history.shape[0]
    z_x = history[:, 0]
    z_y = history[:, 1]
    v_x = 0
    v_y = 0
    for index in range(length_history - 1):
        v_x += z_x[index + 1] - z_x[index]
        v_y += z_y[index + 1] - z_y[index]
    v_x = v_x / (length_history - 1)
    v_y = v_y / (length_history - 1)
    x_x = np.zeros(length_history + 1, np.float32)
    x_y = np.zeros(length_history + 1, np.float32)
    P_x = np.zeros(length_history + 1, np.float32)
    P_y = np.zeros(length_history + 1, np.float32)
    P_vx = np.zeros(length_history + 1, np.float32)
    P_vy = np.zeros(length_history + 1, np.float32)

    # we initialize the uncertainty to one (unit gaussian)
    P_x[0] = 1.0
    P_y[0] = 1.0
    P_vx[0] = 1.0
    P_vy[0] = 1.0
    x_x[0] = z_x[0]
    x_y[0] = z_y[0]

    Q = 0.00001
    R = 0.0001
    K_x = np.zeros(length_history + 1, np.float32)
    K_y = np.zeros(length_history + 1, np.float32)
    K_vx = np.zeros(length_history + 1, np.float32)
    K_vy = np.zeros(length_history + 1, np.float32)
    for k in range(length_history - 1):
        x_x[k + 1] = x_x[k] + v_x
        x_y[k + 1] = x_y[k] + v_y
        P_x[k + 1] = P_x[k] + P_vx[k] + Q
        P_y[k + 1] = P_y[k] + P_vy[k] + Q
        P_vx[k + 1] = P_vx[k] + Q
        P_vy[k + 1] = P_vy[k] + Q
        K_x[k + 1] = P_x[k + 1] / (P_x[k + 1] + R)
        K_y[k + 1] = P_y[k + 1] / (P_y[k + 1] + R)
        x_x[k + 1] = x_x[k + 1] + K_x[k + 1] * (z_x[k + 1] - x_x[k + 1])
        x_y[k + 1] = x_y[k + 1] + K_y[k + 1] * (z_y[k + 1] - x_y[k + 1])
        P_x[k + 1] = P_x[k + 1] - K_x[k + 1] * P_x[k + 1]
        P_y[k + 1] = P_y[k + 1] - K_y[k + 1] * P_y[k + 1]
        K_vx[k + 1] = P_vx[k + 1] / (P_vx[k + 1] + R)
        K_vy[k + 1] = P_vy[k + 1] / (P_vy[k + 1] + R)
        P_vx[k + 1] = P_vx[k + 1] - K_vx[k + 1] * P_vx[k + 1]
        P_vy[k + 1] = P_vy[k + 1] - K_vy[k + 1] * P_vy[k + 1]

    k = k + 1
    x_x[k + 1] = x_x[k] + v_x * prediction_horizon
    x_y[k + 1] = x_y[k] + v_y * prediction_horizon
    P_x[k + 1] = P_x[k] + P_vx[k] * prediction_horizon * prediction_horizon + Q
    P_y[k + 1] = P_y[k] + P_vy[k] * prediction_horizon * prediction_horizon + Q
    P_vx[k + 1] = P_vx[k] + Q
    P_vy[k + 1] = P_vy[k] + Q
    return x_x[k + 1], x_y[k + 1]

# Function to calculate Euclidean distance error
def calculate_error(predicted, true):
    return np.linalg.norm(predicted - true, axis=1)

def find_most_complex_timeframe(positions, history_len=20, prediction_horizon=40, lower_threshold=500, higher_threshold=3000, skip_initial_steps = 30):
    """
    Identifies the most complex timeframe in a trajectory based on prediction error using a Kalman filter.

    This function iterates through time steps in the trajectory, uses a Kalman filter to predict future positions, 
    and calculates the error between predicted and actual positions. The goal is to find a time window where the error 
    falls within a specified range, indicating high trajectory complexity.

    Parameters:
    - positions (numpy.ndarray): Array of shape (N, 2) representing (x, y) positions.
    - history_len (int): Number of past steps to use for predicting future positions.
    - prediction_horizon (int): Number of future steps to predict.
    - lower_threshold (float): Lower bound for error to consider a segment complex.
    - higher_threshold (float): Upper bound for error to consider a segment complex.
    - skip_initial_steps (int): Number of initial frames to skip for better stability.

    Returns:
    - best_scenario_index (int): The index where the most complex timeframe starts.
    - best_error (float): The error associated with the selected complex timeframe.
    """
    
    # Initialize error tracking variables
    best_error = None
    best_scenario_index = None
    highest_error = 0           # Used if no error fits the threshold criteria
    
    # Iterate through the trajectory to find the most complex segment
    for start in range(skip_initial_steps,len(positions) - history_len - prediction_horizon):
        
        # Extract history and true future positions for the current timeframe
        history = positions[start:start + history_len, :2]  # Use only x, y positions
        true_future = positions[start + history_len:start + history_len + prediction_horizon, :2]

        # Initialize predicted future array
        predicted_future = np.zeros_like(true_future)
        
        # Predict the future using a Kalman filter
        for i in range(prediction_horizon):
            predicted_position = estimate_kalman_filter(history, 1) # Predict next step using Kalman filter
            predicted_future[i] = predicted_position
            # Shift history and append the predicted position for the next iteration
            history = np.vstack([history[1:], predicted_position])  # Shift and add new position
        
        # Calculate the error between predicted and actual future positions
        error = calculate_error(predicted_future, true_future)
        total_error_frame = np.sum(error)
        
        # Track the highest error segment, even if it doesn't meet the criteria
        if total_error_frame>highest_error:
            highest_error = total_error_frame
            idx_highest_error = start
        
        # If the error falls within the specified thresholds, select this segment as the best    
        if lower_threshold < total_error_frame < higher_threshold:
            best_error = total_error_frame
            best_scenario_index = start
            break  # Stop searching if a suitable complex segment is found
        
    # If no segment met the threshold, return the segment with the highest error
    if best_scenario_index == None:
        best_scenario_index = idx_highest_error
    
    return best_scenario_index, best_error

def update_scenario_for_timestep(scenario, start_index, length=200):
    """
    Trims and updates a scenario dictionary to include only a specific range of timesteps.

    This function modifies the scenario's metadata and tracks data to focus on a specified 
    subset of the timeline. If the data is shorter than the specified length, padding is applied.

    Parameters:
    - scenario (dict): The original scenario dictionary containing metadata and tracks.
    - start_index (int): The starting index for the desired timeframe.
    - length (int): Number of timesteps to include in the modified scenario (default: 200).

    Returns:
    - dict: The modified scenario dictionary containing the updated timestep range.
    """
    
    # Update scenario length metadata to reflect the trimmed timeframe
    scenario["length"] = length
    
    # ============================
    # Time Array Adjustment
    # ============================
    # Update the time array if present and a NumPy array
    if "ts" in scenario["metadata"] and isinstance(scenario["metadata"]["ts"], np.ndarray):
        # Trim the time array and rebase it to start from zero
        scenario["metadata"]["ts"] = scenario["metadata"]["ts"][start_index:start_index + length] - scenario["metadata"]["ts"][start_index]

        # If the trimmed array is shorter than the specified length, pad the array with evenly spaced values
        if scenario["metadata"]["ts"].shape[0] < length:
            last_value = scenario["metadata"]["ts"][-1] if len(scenario["metadata"]["ts"]) > 0 else 0
            pad_length = length - scenario["metadata"]["ts"].shape[0]
            padding = np.linspace(last_value, last_value + pad_length * 0.1, pad_length)
            scenario["metadata"]["ts"] = np.concatenate((scenario["metadata"]["ts"], padding))
    else:
        # If no valid time array exists, initialize a default time array
        print("Warning: 'ts' field is missing or None; initializing with a default time array.")
        scenario["metadata"]["ts"] = np.linspace(0, (length - 1) * 0.1, length) 
    
    # ============================
    # Update Agent Track Data
    # ============================
    # Trim each agent's track data to the specified range
    for agent_id, agent_data in scenario["tracks"].items():
        # Trim relevant state data fields
        for state_key in ["position", "heading", "velocity", 
                          "valid", "throttle_brake", "steering", 
                          "length", "width", "height","action"
        ]:
            if state_key in agent_data["state"] and isinstance(agent_data["state"][state_key], np.ndarray):
                # Trim data and ensure it is padded if needed
                agent_data["state"][state_key] = agent_data["state"][state_key][start_index:start_index + length]
                agent_data["state"][state_key] = pad_state_array(agent_data["state"][state_key], length)
        
        # Update agent's metadata with the new track length and validity information
        agent_data["metadata"]["track_length"] = length
        valid_entries = agent_data["state"]["valid"]
        agent_data["metadata"]["valid_length"] = int(np.sum(valid_entries))
        agent_data["metadata"]["continuous_valid_length"] = calculate_continuous_valid_length(valid_entries)

    # ============================
    # Update Object Summary Metadata
    # ============================
    # Ensure the object summary metadata is consistent with the updated track data
    if "object_summary" in scenario["metadata"]:
        for object_id, summary_data in scenario["metadata"]["object_summary"].items():
            summary_data["track_length"] = length
            summary_data["valid_length"] = scenario["tracks"][object_id]["metadata"]["valid_length"]
            summary_data["continuous_valid_length"] = scenario["tracks"][object_id]["metadata"]["continuous_valid_length"]

    return scenario

def calculate_continuous_valid_length(valid_entries):
    """
    Calculate the longest continuous stretch of valid entries.

    This function iterates over an array of boolean valid flags and determines
    the longest sequence of consecutive `True` values, representing the longest 
    continuous valid length.

    Parameters:
    - valid_entries (np.ndarray or list): A boolean array where `True` indicates a valid entry.

    Returns:
    - int: The length of the longest continuous sequence of valid entries.
    """
    
    # Initialize counters for the current valid streak and the maximum found
    max_continuous_valid = 0
    current_valid = 0
    
    # Iterate through the valid entries array
    for valid in valid_entries:
        if valid:
            # Increment the current valid streak
            current_valid += 1
            # Update the maximum valid streak if the current one is longer
            max_continuous_valid = max(max_continuous_valid, current_valid)
        else:
            # Reset the current streak when an invalid entry is encountered
            current_valid = 0
            
    # Return the longest continuous sequence of valid entries found
    return max_continuous_valid

def print_structure(d, indent=0):
    """
    Recursively prints the structure of a dictionary.
    :param d: The dictionary to print the structure of
    :param indent: Current indentation level
    """
    for key, value in d.items():
        print("    " * indent + f"- {key}")
        if isinstance(value, dict):  # If the value is a dictionary, go deeper
            print_structure(value, indent + 1)
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
            # If the value is a list of dictionaries, print structure of the first item
            print("    " * (indent + 1) + f"(list of {len(value)} items, showing structure of the first item)")
            print_structure(value[0], indent + 2)
        else:
            print("    " * (indent + 1) + f"(type: {type(value).__name__})")

def pad_state_array(state_array, target_length=60):
    current_length = len(state_array)
    if current_length < target_length:
        padding = np.repeat([state_array[-1]], target_length - current_length, axis=0)
        return np.concatenate([state_array, padding], axis=0)
    return state_array