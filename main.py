"""
main module:
module to run the whole pipleline
"""

# pylint: disable=C0103, C0301

import os
import pathlib
from pathlib import Path

import fire

from src.config import load_config
from src.distance_transform import calculate_distance_tranform
from src.logging_module import mlflow_logging
from src.preprocessing import data_preprocessing_wrapper
from src.profiles import calculate_profile
from src.segmentation import segmentation_wrapper
from src.segmentation_postprocessing import postprocess_masks
from src.utils import set_root_directory

root_directory_path = root_directory_path = set_root_directory()

# Path where the logged mlflow results should be stored
MLFLOW_TRACKING_URI = root_directory_path.joinpath("data/mlruns")

# Name of the experiments folder
MLFLOW_EXPERIMENT_NAME = "demo_results_1"

def main(config_path: pathlib.PosixPath) -> None:

    """

    Main Function for the entire pipeline for automated usage.
    It wraps entire pipeline into one function which can be source from the terminal.
    Entire process is fully automated and uses **config.json** file.
    See [Config file documentation](config.md)

    Parameters
    ----------

    **config_path** : *(pathlib.PosixPath)* Relative path to the config file defining the entire process (Notice the expected folder structure).

    ```bash
    ppdm
    └─ data
       └─ 5IT_STUDY
            └─ **config.json**
            └─ source
                └─raw
                   └─tumor
                   │   └─ 5IT-4X_Ch2_z0300.tiff
                   │   └─    ...
                   │   └─ 5IT-4X_Ch2_z1300.tiff
                   ├─vessel
                   │   └─ 5IT-4X_Ch3_z0300.tiff
                   │   └─    ...
                   │   └─ 5IT-4X_Ch3_z1300.tiff
                   │─virus
                       └─ 5IT-4X_Ch1_z0300.tiff
                       └─    ...
                       └─5IT-4X_Ch1_z1300.tiff
    ```

    Returns
    ------

    Results are dumbed on the disk (segmented masks, distance transform) and save to mlflow (if provided in config).


    Example Usuage
    --------------

    ```python

    >>>conda activate 3d
    >>>python main.py --Config File: data/5IT_DUMMY_STUDY/config.json

    ```

    """

    # Config file
    config_path = root_directory_path.joinpath(config_path)
    experiment = load_config(config_path)

    # script name to log in MLFLOW
    SCRIPT_NAME = "main.py"

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
        force_overwrite=False,
    )

    #######################################
    # # ML-FLOW LOGGING
    # #####################################
    print(MLFLOW_EXPERIMENT_NAME)
    if experiment["mlflow_logging"]:
        mlflow_logging(
            experiment,
            profiles,
            MLFLOW_TRACKING_URI,
            MLFLOW_EXPERIMENT_NAME,
            experiment["mlflow_run_name"],
            SCRIPT_NAME,
        )


if __name__ == "__main__":
    # This is a library, which uses arg-parses under the hood.
    # It handles the parameters parsing and help within the command line interface.
    fire.Fire(main)
