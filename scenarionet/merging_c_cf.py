import os
import pickle
import shutil
import argparse

def update_parent_simulation(parent_file, child_info, output_folder, base_directory):
    """
    Updates a parent simulation file by adding child simulation information.
    
    Parameters:
    - parent_file (str): Path to the parent simulation file.
    - child_info (list): List of dictionaries containing child simulation data.
    - output_folder (str): Path where the updated file will be saved.
    - base_directory (str): Base directory for relative path handling.
    """
    try:
        with open(parent_file, 'rb') as f:
            parent_data = pickle.load(f)
    except (FileNotFoundError, pickle.UnpicklingError, EOFError):
        return  # If the file is not found or corrupted, skip it.

    # Initialize the counterfactual information if not already present
    if "tracks_info_cf" not in parent_data:
        parent_data["tracks_info_cf"] = {}

    # Filter child files that correspond to the current parent file
    parent_base_name = os.path.splitext(os.path.basename(parent_file))[0]
    filtered_child_info = [
        child for child in child_info
        if parent_base_name in os.path.basename(child["file"])
    ]

    # Assign child information to the parent data
    for i, child in enumerate(filtered_child_info):
        parent_data["tracks_info_cf"][i] = child["tracks_info"]
        
    # Save the updated parent data in the specified output folder
    relative_path = os.path.relpath(parent_file, base_directory)
    output_file = os.path.join(output_folder, relative_path)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    try:
        with open(output_file, 'wb') as f:
            pickle.dump(parent_data, f)
    except Exception as e:
        print(f"Error saving updated file {output_file}: {e}")

def copy_additional_files(input_folder, output_folder):
    """
    Copies dataset mapping and summary files from input to output directories.
    """
    for file_name in ["dataset_mapping.pkl", "dataset_summary.pkl"]:
        input_file = os.path.join(input_folder, file_name)
        if os.path.exists(input_file):
            output_file = os.path.join(output_folder, file_name)
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            try:
                shutil.copy2(input_file, output_file)
            except Exception as e:
                print(f"Error copying {file_name}: {e}")

def get_child_simulations(folder):
    """
    Collects child simulation files and their track information from a folder.
    
    Returns:
    - List of dictionaries containing child simulation file paths and track info.
    """
    child_info = []
    for root, _, files in os.walk(folder):
        for file in files:
            if "child" in file.lower() and file.endswith(".pkl"):
                child_path = os.path.join(root, file)
                try:
                    with open(child_path, 'rb') as f:
                        child_data = pickle.load(f)
                    sdc_id = child_data["metadata"]["sdc_id"]
                    tracks_info = child_data["tracks"][sdc_id]
                    child_info.append({"file": child_path, "tracks_info": tracks_info})
                except (KeyError, pickle.UnpicklingError, EOFError):
                    print(f"Error processing child file: {child_path}")
    return child_info

def process_simulation_folders(base_dir, output_dir):
    """
    Processes all simulation folders to update parent simulations with child data.
    """
    for root, _, files in os.walk(base_dir):
        copy_additional_files(root, os.path.join(output_dir, os.path.relpath(root, base_dir)))

        # Filter parent simulation files (non-child .pkl files)
        parent_files = [
            os.path.join(root, file)
            for file in files
            if "child" not in file.lower() and file.endswith(".pkl")
        ]

        # Skip if no parent files found
        if not parent_files:
            continue

        # Get child information from the same directory
        child_info = get_child_simulations(root)
        
        # Update each parent file with corresponding child information
        for parent_file in parent_files:
            update_parent_simulation(parent_file, child_info, output_dir, base_dir)

def update_dataset_summary(summary_file, folder):
    """
    Updates the dataset summary file to include tracks_info_cf for each parent simulation.
    """
    try:
        with open(summary_file, 'rb') as f:
            summary_data = pickle.load(f)
    except (FileNotFoundError, pickle.UnpicklingError, EOFError):
        return

    updated_summary = {}
    for key, value in summary_data.items():
        if "child" in key.lower():
            continue
        
        # Load the parent file and extract counterfactual information
        parent_file = os.path.join(folder, key)
        try:
            with open(parent_file, 'rb') as f:
                parent_data = pickle.load(f)
            tracks_info_cf = parent_data["tracks_info_cf"]
        except (FileNotFoundError, pickle.UnpicklingError, EOFError):
            tracks_info_cf = {}

        value["tracks_info_cf"] = tracks_info_cf
        updated_summary[key] = value
        
    if "tracks_info_cf" in updated_summary:
        del updated_summary["tracks_info_cf"]

    # Save the updated summary file
    try:
        with open(summary_file, 'wb') as f:
            pickle.dump(updated_summary, f)
    except Exception as e:
        print(f"Error saving updated summary file: {summary_file}, {e}")

def process_all_summaries(base_dir):
    """
    Iterates through all subdirectories to update dataset summaries.
    """
    for root, _, files in os.walk(base_dir):
        if "dataset_summary.pkl" in files:
            summary_file = os.path.join(root, "dataset_summary.pkl")
            update_dataset_summary(summary_file, root)

def update_dataset_mapping(mapping_file):
    """
    Cleans the dataset mapping file by removing child entries.
    """
    try:
        with open(mapping_file, 'rb') as f:
            mapping_data = pickle.load(f)
    except (FileNotFoundError, pickle.UnpicklingError, EOFError):
        return

    # Remove child entries and unnecessary keys
    updated_mapping = {key: value for key, value in mapping_data.items() if "child" not in key.lower() and key != "tracks_info_cf"}

    try:
        with open(mapping_file, 'wb') as f:
            pickle.dump(updated_mapping, f)
    except Exception as e:
        print(f"Error saving updated mapping file: {mapping_file}, {e}")

def process_dataset_mappings(base_dir):
    """
    Updates the dataset mapping file in the base directory and its subfolders.
    """
    main_mapping_file = os.path.join(base_dir, "dataset_mapping.pkl")
    if os.path.isfile(main_mapping_file):
        update_dataset_mapping(main_mapping_file)

    for subfolder in os.listdir(base_dir):
        subfolder_path = os.path.join(base_dir, subfolder)
        if os.path.isdir(subfolder_path):
            subfolder_mapping_file = os.path.join(subfolder_path, "dataset_mapping.pkl")
            if os.path.isfile(subfolder_mapping_file):
                update_dataset_mapping(subfolder_mapping_file)
                
def update_main_dataset_summary(base_dir):
    """
    Updates the main dataset summary by removing child entries and including counterfactual data.
    """
    summary_file = os.path.join(base_dir, "dataset_summary.pkl")
    try:
        with open(summary_file, 'rb') as f:
            summary_data = pickle.load(f)
    except (FileNotFoundError, pickle.UnpicklingError, EOFError):
        return

    updated_summary = {}

    for key, value in summary_data.items():
        if "child" in key.lower():
            continue

        parent_file = None
        for subfolder in os.listdir(base_dir):
            subfolder_path = os.path.join(base_dir, subfolder)
            if os.path.isdir(subfolder_path):
                potential_file = os.path.join(subfolder_path, key)
                if os.path.isfile(potential_file):
                    parent_file = potential_file
                    break

        if not parent_file:
            continue

        try:
            with open(parent_file, 'rb') as f:
                parent_data = pickle.load(f)
            tracks_info_cf = parent_data.get("tracks_info_cf", {})
        except (FileNotFoundError, pickle.UnpicklingError, EOFError):
            tracks_info_cf = {}

        value["tracks_info_cf"] = tracks_info_cf
        updated_summary[key] = value

    try:
        with open(summary_file, 'wb') as f:
            pickle.dump(updated_summary, f)
    except Exception as e:
        print(f"Error saving updated main summary file: {summary_file}, {e}")

def main():
    parser = argparse.ArgumentParser(description="Process simulation dataset.")
    parser.add_argument("--base_directory", required=True, help="Base directory for original simulations.")
    parser.add_argument("--output_directory", required=True, help="Directory to save the updated dataset.")
    args = parser.parse_args()

    base_directory = args.base_directory
    output_directory = args.output_directory

    process_simulation_folders(base_directory, output_directory)
    process_all_summaries(output_directory)
    update_main_dataset_summary(output_directory)
    process_dataset_mappings(output_directory)

if __name__ == "__main__":
    main()
