"""
utils module:
default utilitis helper functions
"""

# pylint: disable=E1101, R1705, R0124, R1710, C0301, C0103


import datetime as dt
import inspect
import pathlib
import subprocess
import os
from collections import defaultdict
from functools import wraps
from pathlib import Path
from typing import List


import cv2
import numpy as np
import pandas as pd


def create_relative_path(
    data_path: pathlib.PosixPath,
    folder_path: str,
    folder_name: str,
    _infer_root_based_on: str = "source",
    _keyword: str = "___",
) -> pathlib.PosixPath:
    """
    Infer root and create new folder.
    """
    data_root_path = Path(
        *data_path.parts[: data_path.parts.index(_infer_root_based_on)]
    )
    if _keyword in data_path.parts[-1]:
        folder_parent_name = data_path.parts[-2]
    else:
        folder_parent_name = data_path.parts[-1]
    created_folder = (
        data_root_path / folder_path / folder_parent_name / folder_name
    )
    created_folder.mkdir(exist_ok=True, parents=True)
    return created_folder


def create_preprocessing_relat_directory(
    input_directory: pathlib.PosixPath,
) -> pathlib.PosixPath:

    """
    Infer root and create a new folder for preprocessing
    """

    output_directory = input_directory.parent.parent.joinpath(
        "transformed", "np_and_resized", f"{input_directory.stem}",
    )

    return output_directory


def join_keys_and_values_to_list(
    dictionary: dict, join_with_character: str = "-",
) -> List[str]:
    """
    Join keys and values from dictionary to list.
    """
    keys_zipped_with_values = zip(
        list(dictionary.keys()), list(dictionary.values())
    )
    keys_values = []
    for key, value in keys_zipped_with_values:
        # if value is a path, take only it's name without suffix
        if Path(str(value)).exists():
            value = Path(value).stem
        keys_values.append(
            join_with_character.join([str(key), str(value).replace(".", "_")])
        )
    return keys_values


def join_to_string(items: List[str], join_with_character: str = "___") -> str:
    """
    Join items to one string.
    """
    string = join_with_character.join(items)
    return string


class PrettyDefaultDict(defaultdict):
    """
    Defaultdict with nicer printing.
    """

    __repr__ = dict.__repr__


def nested_dict():
    """
    Create recursive nested dict.
    """
    return PrettyDefaultDict(nested_dict)


def pathlib_to_string(dictionary) -> dict:
    """
    Convert pathlib objects to string in nested dictionary.
    """
    if not isinstance(dictionary, dict):
        if isinstance(dictionary, pathlib.PosixPath):
            return str(dictionary)
        else:
            return dictionary
    return {k: pathlib_to_string(v) for k, v in dictionary.items()}


def flatten(dictionary: dict) -> dict:
    """
    Flatten dictionary in a way that leafs are the only values and all
    parent keys are joined together.
    """
    out = {}
    for key, val in dictionary.items():
        if isinstance(val, dict):
            val = [val]
        if isinstance(val, list):
            for subdict in val:
                deeper = flatten(subdict).items()
                out.update({key + "_" + key2: val2 for key2, val2 in deeper})
        else:
            out[key] = val
    return out


def log_step(func_in=None, *, show_params=False):
    """
    Decorator to print function call details.
    This includes parameters names and effective values.
    """

    def stopwatch(f):
        @wraps(f)
        def func(*args, **kwargs):
            print("\n")
            print(f"running {func.__name__}")
            print("\n")
            tic = dt.datetime.now()
            result = f(*args, **kwargs)
            time_taken = str(dt.datetime.now() - tic)
            print(f"function {func.__name__} took {time_taken}s")

            if show_params:
                func_args = (
                    inspect.signature(func).bind(*args, **kwargs).arguments
                )
                func_args_str = ",".join(
                    map("{0[0]} = {0[1]!r}".format, func_args.items())
                )
                func_args_str = func_args_str.replace(",", ",\n").replace(
                    "{", "\n{"
                )
                print(
                    f"\n used arguments: \n==================\n{func_args_str} \n==================\n"
                )
            return result

        return func

    # This is where the "magic" happens.
    if func_in is None:
        return stopwatch
    else:
        return stopwatch(func_in)
        
def set_root_directory() -> None:
    """
    This function sets root path to script which calls it.

    Returns:
        pathlib.PosixPath: relative path to the root directory
    """    
    root_directory_path = Path(os.path.dirname(os.path.realpath("__file__")))
    print(f"setting root directory to: {root_directory_path}")
    os.chdir(root_directory_path)
    return root_directory_path


def check_content_of_two_directories(
    input_directory, output_directory
) -> bool:
    """
    ----------
    This function check if both input and output directories contain the same number of file

    ----------
    Args:
        input_directory (pathlib):
        output_directory (pathlib):

    Returns:
        bool: True if number of files within both directories equal
    """
    if input_directory.exists() and output_directory.exists():
        file_names_1 = [
            file.stem for file in sorted(list(input_directory.glob("*")))
        ]
        file_names_2 = [
            file.stem for file in sorted(list(output_directory.glob("*")))
        ]

        if len(file_names_1) == 0:
            raise AttributeError(
                str(f"Input directory is empty, check content of folder {input_directory}")
            )

        if len(file_names_1) != len(file_names_2):
            print("Number of files with the folders does not match")
            return False

        elif (len(file_names_1) == len(file_names_2)) and (
            file_names_1 == file_names_2
        ) is False:
            print("Both Folders contain different files")
            return False

        elif (len(file_names_1) == len(file_names_2)) and (
            file_names_1 == file_names_2
        ) is True:
            return True
    else:
        return False


def remove_content(directory, remove_directory=False) -> None:
    """
    ----------
    Given a directory path, this function will iterative remove
    the entire concent of a given directory
    including all files as well as folders found within the directory.

    ----------
    Args:

        directory (pathlib): relative path to the directory
        remove_directory (bool, optional): Once all files within directory are removed,
        should the directory folder itself be also removed?.

        Defaults to False.
    """
    # Folder Content of a given directory
    content = sorted(list(directory.glob("*")))
    print("Deleting {} files in {}".format(len(content), directory.stem))
    for sub in directory.iterdir():
        if sub.is_dir():
            remove_content(sub)
        else:
            sub.unlink()
    if remove_directory:
        directory.rmdir()
    print("Done!")


def _output_to_list(string):
    """
    Decode and split.
    """
    return string.decode("ascii").split("\n")


def select_avail_gpu():
    """
    Get index of available GPU.
    """
    process_free_info = ""
    for i in range(8):
        check_all_gpu_statuses = (
            "nvidia-smi -i %s --query-compute-apps=pid --format=csv,noheader"
            % str(i)
        )
        process_free_info = _output_to_list(
            subprocess.check_output(check_all_gpu_statuses.split())
        )[0]
        if process_free_info == "":
            print("Using gpu ", str(i))
            return i
    if process_free_info != "":
        raise IOError("NO FREE GPU")


def custom_rescale(img: np.ndarray, scale: float) -> np.ndarray:
    """
    Rescale image.
    """
    height, width = np.array(np.array(img.shape) * scale).astype(int)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
    return img


@log_step
def rescale_column_pandas(
    data_frame: pd.DataFrame, column: str, scaling_factor: float
) -> pd.DataFrame:
    """
    Rescale column in Pandas DataFrame.
    """
    data_frame = data_frame.copy()
    data_frame[column] = data_frame[column] * scaling_factor
    return data_frame


# content of test_sample.py
def inc(number_to_add):
    """[summary]

    Args:
        x ([type]): [description]

    Returns:
        [type]: [description]
    """
    print(number_to_add)
    print("hey")
    return number_to_add + 1
