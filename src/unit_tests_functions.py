"""
This script contains unit test functions
"""

# LINKS: https://stackoverflow.com/questions/18011902/pass-a-parameter-to-a-fixture-function
# https://www.guru99.com/pytest-tutorial.html#10

import json
import warnings

import numpy as np
import pytest
from skimage.io import imsave

warnings.filterwarnings("ignore")


def create_json(path_file, value):
    """
    this function save json file to a disk
    """
    with open(path_file, "w") as file_to_save:
        json.dump({"number": value}, file_to_save)


@pytest.fixture(name="images")
def create_images(tmp_path):
    """
    This function creates artifical folder with n_files_within_folders within it.
    Number of files within both folders is similiar
    """
    for file in range(1, 20):
        path_of_file = tmp_path / "data" / "images"
        path_of_file.mkdir(exist_ok=True, parents=True)
        file_path = path_of_file / f"img_{file}.tiff"
        img = np.zeros([100, 100, 1], dtype=np.uint8)
        img.fill(255)
        imsave(file_path, img)

    return path_of_file


@pytest.fixture(name="even")
def create_data_even(tmp_path, n_files_within_folders=5):
    """
    This function creates artifical folder with n_files_within_folders within it.
    Number of files within both folders is similiar
    """
    paths = []
    for folder in [1, 2]:
        for file in range(1, n_files_within_folders):
            path_of_file = tmp_path / "data" / f"folder-{folder}"
            path_of_file.mkdir(exist_ok=True, parents=True)
            file_path = path_of_file / f"file-{file}.json"
            create_json(file_path, file)
        paths.append(path_of_file)

    return {"input_directory": paths[0], "output_directory": paths[1]}


@pytest.fixture(name="un_even")
def create_data_un_even(tmp_path, n_files_within_folders=5):
    """
    This function creates artifical folder with n_files_within_folders within it.
    Number of files within both folders is NOT similiar
    """
    paths = []
    for folder in [1, 2]:
        for file in range(1, n_files_within_folders):
            path_of_file = tmp_path / "data" / f"folder-{folder}"
            path_of_file.mkdir(exist_ok=True, parents=True)
            file_path = path_of_file / f"file-{file}.json"
            create_json(file_path, file)
        paths.append(path_of_file)
        n_files_within_folders *= 2

    return {"input_directory": paths[0], "output_directory": paths[1]}


@pytest.fixture(name="empty")
def create_data_empty(tmp_path):
    """
    This function creates artifical folder with n_files_within_folders within it.
    Number of files within both folders is NOT similiar
    """
    paths = []
    for folder in [1, 2]:
        path_of_file = tmp_path / "data" / f"folder-{folder}"
        path_of_file.mkdir(exist_ok=True, parents=True)
        paths.append(path_of_file)

    return {"input_directory": paths[0], "output_directory": paths[1]}
