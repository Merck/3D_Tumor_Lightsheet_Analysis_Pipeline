"""
## Preprocessing module:
### Converting Images to numpy form and resizing
#### This module consists of two main functions:

* convert_images_to_numpy_format
* data_preprocessing_wrapper

"""

# pylint: disable=C0301

import pathlib
from functools import partial

import numpy as np
from fastcore.parallel import parallel
from skimage.io import imread
from skimage.transform import rescale

from src.utils import (
    check_content_of_two_directories,
    create_preprocessing_relat_directory,
    log_step,
    nested_dict,
    remove_content,
)


@log_step
def data_preprocessing_wrapper(data: dict) -> dict:

    """
    High level function that covers preprocessing for all data (blood vessels, tumors, virus).
    It used the convert_images_to_numpy_format function and applies it three times for each channel.
    If you want to preprocess just one channel (e.g. only tumors) use the convert_images_to_numpy_format function.


    Parameters
    ----------

    **data**: *(dict)* containing keys (names of the channels) and values (relative paths to it).

    Returns
    ------
    **preprocessed_data**: *(dict) outputs three relative paths for the folders with preprocessed results (tranfered and converted images)

    Results are dumbed on the disk.

    Example Usuage
    --------------
    ```python

    >>>from src.preprocessing import data_preprocessing_wrapper
    >>>study_paths = {'vessel': Path('ppdm/data/5IT_DUMMY_STUDY/source/raw/vessel'),
     'tumor': Path('ppdm/data/5IT_DUMMY_STUDY/source/raw/tumor'),
     'virus': Path('ppdm/data/5IT_DUMMY_STUDY/source/raw/virus')}

    >>>data_preprocessing_wrapper(study_paths)

    >>>output:
    defaultdict(<function src.utils.nested_dict()>,
            {'vessel': Path('ppdm/data/5IT_DUMMY_STUDY/source/transformed/np_and_resized/vessel'),
             'tumor': Path('ppdm/data/5IT_DUMMY_STUDY/source/transformed/np_and_resized/tumor'),
             'virus': Path('ppdm/data/5IT_DUMMY_STUDY/source/transformed/np_and_resized/virus')})

    ```
    """

    preprocessed_data = nested_dict()

    out_path = convert_images_to_numpy_format(data["vessel"])
    preprocessed_data["vessel"] = out_path

    # tumor
    out_path = convert_images_to_numpy_format(data["tumor"])
    preprocessed_data["tumor"] = out_path

    # virus
    out_path = convert_images_to_numpy_format(data["virus"])
    preprocessed_data["virus"] = out_path

    return preprocessed_data


def _create_numpy_formats_of_input_img(img_path, output_directory) -> None:
    """
    ----------
    This function convert images within given input directory into
    numpy format and reduce them in size.
    All numpy images are saved into output_directory.

    ----------
    Args:
        input_directory (pathlib):
        output_directory ([pathlib):
    """
    img_i = imread(img_path)
    img_dtype = img_i.dtype
    img_i = rescale(
        # 1.8/4 because 4 microns in z-layers are give by microscope, and 1.8 in x and y-axis are also given.
        # Values here are hard-coded if they change microscope, we may need to change the values here
        img_i,
        scale=1.8 / 4,
        anti_aliasing=True,
        preserve_range=True,
    )
    img_i = img_i.astype(img_dtype)
    np.save(output_directory / f"{img_path.stem}.npy", img_i)


def convert_images_to_numpy_format(
    input_directory: pathlib.PosixPath,
) -> pathlib.PosixPath:
    """

    wrapper function that reduces images in the input folder and converts content of folder to numpy format in parallel (faster)


    Parameters
    ----------

    **input_directory**: *(pathlib.PosixPath object)* relative path to the images which should be preprocessed


    Returns
    ------

    **output_directory**: *(pathlib.PosixPath object)* one relative paths for the folder with preprocessed results (tranfered and converted images)


    Example Usuage
    --------------
    ```python
    >>>from src.preprocessing import convert_images_to_numpy_format
    >>>tumor_folder = Path('ppdm/data/5IT_DUMMY_STUDY/source/raw/tumor')
    >>>convert_images_to_numpy_format(tumor_folder)

    >>>output:
    Path('ppdm/data/5IT_DUMMY_STUDY/source/transformed/np_and_resized/tumor')
    ```

    """
    output_directory = create_preprocessing_relat_directory(input_directory)

    output_directory.mkdir(parents=True, exist_ok=True)

    directory_ok = check_content_of_two_directories(
        input_directory, output_directory
    )

    if directory_ok is False:
        if output_directory.exists():
            remove_content(output_directory)

        parallel_create_numpy_formats_of_input_img = partial(
            _create_numpy_formats_of_input_img,
            output_directory=output_directory,
        )

        parallel(
            parallel_create_numpy_formats_of_input_img,
            sorted(list(input_directory.glob("*.tiff"))),
            n_workers=12,
            progress=True,
            threadpool=True,
        )

    return output_directory
