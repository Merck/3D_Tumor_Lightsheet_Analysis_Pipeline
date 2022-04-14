"""
test_preprocessing:
This module contains functions that are used for the unit-testing
"""

# pylint: disable=W0611

import warnings
from pathlib import PosixPath

import pytest

from src.preprocessing import (
    check_content_of_two_directories,
    convert_images_to_numpy_format,
)
from src.unit_tests_functions import (
    create_data_empty,
    create_data_even,
    create_data_un_even,
    create_images,
    create_json,
)

warnings.filterwarnings("ignore")


def test_output_format_of_convert_images_to_numpy_format_output(images):
    """
    Does the function return Pathlib object?
    """
    data_test = images
    output = convert_images_to_numpy_format(data_test)
    assert isinstance(output, PosixPath)


def test_check_content_of_two_empty_directories(empty):
    """
    When we add an empty input directory, AttributeError shall arise
    """
    folder_paths = empty
    with pytest.raises(AttributeError):
        check_content_of_two_directories(**folder_paths)


def test_check_content_of_two_even_directories(even):
    """
    When we create 2 folder with same number of files inside them, the function should pass
    """
    # Create artificial files
    folder_paths = even
    assert check_content_of_two_directories(**folder_paths) is True


def test_check_content_of_two_uneven_directories(un_even):
    """
    When we create 2 folder with different number of elements, function should return FALSE
    """
    # Create artificial files
    folder_paths = un_even
    assert check_content_of_two_directories(**folder_paths) is False
