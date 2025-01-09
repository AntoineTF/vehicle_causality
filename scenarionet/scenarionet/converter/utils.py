import ast
import copy
import inspect
import logging
import math
import multiprocessing
import os
import pickle
import shutil
from functools import partial

import numpy as np
import psutil
import tqdm
from metadrive.scenario import ScenarioDescription as SD

from scenarionet.builder.utils import merge_database
from scenarionet.common_utils import save_summary_and_mapping
from scenarionet.converter.pg.utils import convert_pg_scenario, make_env

logger = logging.getLogger(__file__)


def single_worker_preprocess(x, worker_index):
    """
    All scenarios passed to write_to_directory_single_worker will be preprocessed. The input is expected to be a list.
    The output should be a list too. The element in the second list will be processed by convertors. By default, you
    don't need to provide this processor. We override it for waymo convertor to release the memory in time.
    :param x: input
    :param worker_index: worker_index, useful for logging
    :return: input
    """
    return x


def nuplan_to_metadrive_vector(vector, nuplan_center=(0, 0)):
    "All vec in nuplan should be centered in (0,0) to avoid numerical explosion"
    vector = np.array(vector)
    vector -= np.asarray(nuplan_center)
    return vector


def compute_angular_velocity(initial_heading, final_heading, dt):
    """
    Calculate the angular velocity between two headings given in radians.

    Parameters:
    initial_heading (float): The initial heading in radians.
    final_heading (float): The final heading in radians.
    dt (float): The time interval between the two headings in seconds.

    Returns:
    float: The angular velocity in radians per second.
    """

    # Calculate the difference in headings
    delta_heading = final_heading - initial_heading

    # Adjust the delta_heading to be in the range (-π, π]
    delta_heading = (delta_heading + math.pi) % (2 * math.pi) - math.pi

    # Compute the angular velocity
    angular_vel = delta_heading / dt

    return angular_vel


def mph_to_kmh(speed_in_mph: float):
    speed_in_kmh = speed_in_mph * 1.609344
    return speed_in_kmh


def contains_explicit_return(f):
    return any(isinstance(node, ast.Return) for node in ast.walk(ast.parse(inspect.getsource(f))))


def write_to_directory(
    convert_func,
    scenarios,
    output_path,
    dataset_version,
    dataset_name,
    overwrite=False,
    num_workers=8,
    preprocess=single_worker_preprocess,
    **kwargs
):
    # make sure dir not exist
    kwargs_for_workers = [{} for _ in range(num_workers)]
    for key, value in kwargs.items():
        for i in range(num_workers):
            kwargs_for_workers[i][key] = value[i]

    save_path = copy.deepcopy(output_path)
    if os.path.exists(output_path):
        if not overwrite:
            raise ValueError(
                "Directory {} already exists! Abort. "
                "\n Try setting overwrite=True or adding --overwrite".format(output_path)
            )
        else:
            shutil.rmtree(output_path)
    os.makedirs(save_path, exist_ok=False)

    basename = os.path.basename(output_path)
    # dir = os.path.dirname(output_path)
    for i in range(num_workers):
        subdir = os.path.join(output_path, "{}_{}".format(basename, str(i)))
        if os.path.exists(subdir):
            if not overwrite:
                raise ValueError(
                    "Directory {} already exists! Abort. "
                    "\n Try setting overwrite=True or adding --overwrite".format(subdir)
                )
    # get arguments for workers
    num_files = len(scenarios)
    if num_files < num_workers:
        # single process
        logger.info("Use one worker, as num_scenario < num_workers:")
        num_workers = 1

    argument_list = []
    output_pathes = []
    num_files_each_worker = int(num_files // num_workers)
    for i in range(num_workers):
        if i == num_workers - 1:
            end_idx = num_files
        else:
            end_idx = (i + 1) * num_files_each_worker
        subdir = os.path.join(output_path, "{}_{}".format(basename, str(i)))
        output_pathes.append(subdir)
        argument_list.append([scenarios[i * num_files_each_worker:end_idx], kwargs_for_workers[i], i, subdir])

    # prefill arguments
    func = partial(
        writing_to_directory_wrapper,
        convert_func=convert_func,
        dataset_version=dataset_version,
        dataset_name=dataset_name,
        preprocess=preprocess,
        overwrite=overwrite
    )

    # Run, workers and process result from worker
    with multiprocessing.Pool(num_workers, maxtasksperchild=10) as p:
        ret = list(p.imap(func, argument_list))
        # call ret to block the process
    merge_database(save_path, *output_pathes, exist_ok=True, overwrite=False, try_generate_missing_file=False)


def writing_to_directory_wrapper(
    args, convert_func, dataset_version, dataset_name, overwrite=False, preprocess=single_worker_preprocess
):
    return write_to_directory_single_worker(
        convert_func=convert_func,
        scenarios=args[0],
        output_path=args[3],
        dataset_version=dataset_version,
        dataset_name=dataset_name,
        preprocess=preprocess,
        overwrite=overwrite,
        worker_index=args[2],
        **args[1]
    )


def write_to_directory_single_worker(
    convert_func,
    scenarios,
    output_path,
    dataset_version,
    dataset_name,
    worker_index=0,
    overwrite=False,
    report_memory_freq=None,
    preprocess=single_worker_preprocess,
    **kwargs
):
    """
    Convert a batch of scenarios and save each one to a specified directory.
    
    Parameters:
    - convert_func (callable): Function used to convert each scenario (e.g., `convert_pg_scenario`).
    - scenarios (list): List of scenario identifiers to process.
    - output_path (str): Directory where the converted scenarios will be saved.
    - dataset_version (str): Version identifier for the dataset (used for file naming).
    - dataset_name (str): Name of the dataset (used for file naming).
    - worker_index (int): Index of the worker processing the scenarios, used for parallelization.
    - overwrite (bool): If True, overwrites existing output directories and data.
    - report_memory_freq (int, optional): Frequency for reporting memory usage during processing.
    - preprocess (callable): Function to preprocess scenarios before conversion.
    - kwargs: Additional arguments passed to the `convert_func`.
    """
    
    # Ensure the conversion function returns metadata (prevents silent errors)
    if not contains_explicit_return(convert_func):
        raise RuntimeError("The convert function should return a metadata dict")

    # Ensure the dataset version is consistent and avoid conflicts
    if "version" in kwargs:
        kwargs.pop("version")
        logger.info("the specified version in kwargs is replaced by argument: 'dataset_version'")

    # Apply the preprocessing step to prepare scenarios if provided
    scenarios = preprocess(scenarios, worker_index)

    # Prepare the output directory path with a temporary naming convention to avoid conflicts
    save_path = copy.deepcopy(output_path)
    output_path = output_path + "_tmp"
    
    # If the directory already exists and overwrite is enabled, remove it
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=False)

    # Handle existing output directories
    delay_remove = None
    if os.path.exists(save_path):
        if overwrite:
            delay_remove = save_path
        else:
            raise ValueError("Directory already exists! Abort." "\n Try setting overwrite=True or using --overwrite")
    
    # Prepare metadata paths
    summary_file = SD.DATASET.SUMMARY_FILE
    mapping_file = SD.DATASET.MAPPING_FILE
    summary_file_path = os.path.join(output_path, summary_file)
    mapping_file_path = os.path.join(output_path, mapping_file)

    # Initialize summary and mapping dictionaries
    summary = {}
    mapping = {}

    # For PG scenarios, set the environment
    if convert_func is convert_pg_scenario:
        kwargs["env"] = make_env(start_index=scenarios[0], num_scenarios=len(scenarios))

    count = 0
    
    # Main loop to process each scenario
    for scenario in scenarios:
        
        # Convert the current scenario using the provided conversion function
        sd_scenario = convert_func(scenario, dataset_version, **kwargs)
        
        # If the conversion returns valid data, proceed with storing the scenario
        if sd_scenario is not None:
            
            # Extract the unique scenario identifier for file naming
            scenario_id = sd_scenario[SD.ID]
            export_file_name = SD.get_export_file_name(dataset_name, dataset_version, scenario_id)

            # Update the summary and mapping dictionaries with scenario metadata
            if hasattr(SD, "update_summaries"):
                SD.update_summaries(sd_scenario)
            else:
                raise ValueError("Please update MetaDrive to latest version.")

            # Check for overwriting conflicts
            if export_file_name in summary:
                logger.warning("Scenario {} already exists and will be overwritten!".format(export_file_name))
            
            # Save the metadata and empty mapping for the current scenario
            summary[export_file_name] = copy.deepcopy(sd_scenario[SD.METADATA])
            mapping[export_file_name] = ""  # in the same dir

            # Convert the scenario to a dictionary format for easier storage
            sd_scenario = sd_scenario.to_dict()
            
            # Perform a sanity check on the scenario data for consistency
            SD.sanity_check(sd_scenario, check_self_type=True)

            # Serialize the scenario data using pickle and save it to the output path
            p = os.path.join(output_path, export_file_name)
            with open(p, "wb") as f:
                pickle.dump(sd_scenario, f)

            # Periodically report memory usage for monitoring large dataset processing
            if report_memory_freq is not None and (count) % report_memory_freq == 0:
                print("Current Memory: {}".format(process_memory()))
                
            count += 1

            # Log progress after every 500 processed scenarios
            if count % 500 == 0:
                logger.info(f"Worker {worker_index} has processed {count} scenarios.")

    # Save the collected metadata and mapping information
    save_summary_and_mapping(summary_file_path, mapping_file_path, summary, mapping)

    # If overwriting, remove the original directory before renaming the temporary folder
    if delay_remove is not None:
        assert delay_remove == save_path
        shutil.rmtree(delay_remove)
    
    # Rename the temporary folder to the final directory name
    os.rename(output_path, save_path)

    # Log completion message
    logger.info(f"Worker {worker_index} finished! Files are saved at: {save_path}")


def process_memory():
    """
    Get the current memory usage of the running process in MB.
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024  # Convert bytes to megabytes (MB)
