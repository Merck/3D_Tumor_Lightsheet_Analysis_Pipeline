"""
main module:
module to run the whole pipleline
"""

import os
from pathlib import Path

from src.config import load_config
from src.distance_transform import calculate_distance_tranform
from src.logging_module import mlflow_logging
from src.preprocessing import data_preprocessing_wrapper
from src.profiles import calculate_profile, vizualize_profile
from src.segmentation import segmentation_wrapper
from src.segmentation_postprocessing import postprocess_masks
from src.utils import set_root_directory

root_directory_path = set_root_directory()

# Config file
config_path = root_directory_path.joinpath("data/5IT_DUMMY_STUDY/config.json")
experiment = load_config(config_path)

MLFLOW_TRACKING_URI = root_directory_path.joinpath("data/mlruns")
MLFLOW_EXPERIMENT_NAME = "experiments"
SCRIPT_NAME = "master_script.py"

#######################################
# # DATA PREPROCESSING
# #####################################

experiment["data"]["source"]["transformed"] = data_preprocessing_wrapper(
    experiment["data"]["source"]["raw"]
)

#######################################
# # SEGMENTATION
# #####################################

# segmentation of blood vessels
out_path = segmentation_wrapper(
    experiment["data"]["source"]["transformed"]["vessel"],
    **experiment["segmentation_method_vessel"],
)
experiment["data"]["results"]["segmentation"]["vessel"] = out_path


# segmentation of tumors
out_path = segmentation_wrapper(
    experiment["data"]["source"]["transformed"]["tumor"],
    **experiment["segmentation_method_tumor"],
)
experiment["data"]["results"]["segmentation"]["tumor"] = out_path

# # postprocessing tumor masks
out_path = postprocess_masks(
    experiment["data"]["results"]["segmentation"]["tumor"],
    **experiment["segmentation_postprocessing_tumor"],
)
experiment["data"]["results"]["segmentation_postprocessing"][
    "tumor"
] = out_path

#######################################
# # DISTANCE TRANSFORM
# #####################################

out_path = calculate_distance_tranform(
    experiment["data"]["results"]["segmentation"]["vessel"],
    **experiment["distance_tranform"]["method_parameters"],
)
experiment["data"]["results"]["distance_transform"]["vessel"] = out_path

#######################################
# # PROFILE
# #####################################

# calculating the final profile
profiles = calculate_profile(
    experiment["data"]["source"]["transformed"]["virus"],
    experiment["data"]["results"]["distance_transform"]["vessel"],
    experiment["data"]["results"]["segmentation_postprocessing"]["tumor"],
    experiment["pixels_to_microns"],
    force_overwrite=True,
)

""
profile_all = vizualize_profile(profiles["all"])
profile_core = vizualize_profile(profiles["core"])
profile_periphery = vizualize_profile(profiles["periphery"])

#######################################
# # ML-FLOW LOGGING
# #####################################

if experiment["mlflow_logging"]:
    mlflow_logging(
        experiment,
        profiles,
        MLFLOW_TRACKING_URI,
        MLFLOW_EXPERIMENT_NAME,
        experiment["mlflow_run_name"],
        SCRIPT_NAME,
    )
