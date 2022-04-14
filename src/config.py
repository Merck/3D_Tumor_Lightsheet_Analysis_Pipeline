"""
Config module gathering function to load, prepare and validate config file.
"""

# pylint: disable=C0301

import json
import pathlib
from pathlib import Path
from typing import Union

import pydantic

from src.config_validation_templates import ConfigTemplate
from src.utils import nested_dict


def load_config(path: Union[pathlib.PosixPath, str]) -> dict:
    """
    Load config from json file.
    """

    with open(path) as json_file:
        config = json.load(json_file)
    config["data"] = get_data_paths(path)
    validate(config, ConfigTemplate)

    # Default path for pre-trained unet method
    if config["segmentation_method_vessel"]["method"] == "unet":
        config["segmentation_method_vessel"]["method_parameters"] = {
            "model_file": "pretrained_models/unet_train_full_5IT_7IV_8IV.pt"
        }

    # Default path for pre-trained random forest
    elif config["segmentation_method_vessel"]["method"] == "random_forest":
        config["segmentation_method_vessel"]["method_parameters"] = {
            "model_file": "pretrained_models/rf_model.joblib"
        }

    return config


def validate(config: dict, template: pydantic.main.ModelMetaclass) -> str:
    """
    Validate config dictionary to match given template using Pydantic library.
    """
    return template(**config).dict()


def get_data_paths(path: Union[pathlib.PosixPath, str]) -> dict:
    """
    Get data paths relative to config path.
    """
    root_directory = Path(path).parent
    data_paths_dict = nested_dict()
    data_paths_dict["source"]["raw"]["vessel"] = root_directory.joinpath(
        "source/raw/vessel"
    )
    data_paths_dict["source"]["raw"]["tumor"] = root_directory.joinpath(
        "source/raw/tumor"
    )
    data_paths_dict["source"]["raw"]["virus"] = root_directory.joinpath(
        "source/raw/virus"
    )
    return data_paths_dict
