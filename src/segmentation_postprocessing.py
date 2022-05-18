"""
## segmentation postprocessing module:
### module for certain masks postprocessing operation
### Implemented Operations: Reducing Tumor Borders and Filling holes
#TODO: is it discussed in the paper?
#### This module consists of two main functions:

* postprocess_masks
* split_tumor_into_core_and_periphery

"""


# pylint: disable=R1721, R0914, C0301

import pathlib
import sys
from collections import Counter
from functools import partial
from typing import Dict, List

import cv2
import edt
import numpy as np
from fastcore.parallel import parallel
from scipy.ndimage import zoom
from skimage.morphology import disk, label
from skimage.morphology.grey import dilation

from src.utils import (
    check_content_of_two_directories,
    create_relative_path,
    custom_rescale,
    join_keys_and_values_to_list,
    join_to_string,
    log_step,
    remove_content,
)


@log_step
def postprocess_masks(
    data_path: pathlib.PosixPath, method: str, method_parameters: Dict
) -> pathlib.PosixPath:
    """
    Wrapper function, which handles the masks postprocessing within the pipeline.
    For now only used for the tumor masks postprocessing to split masks into core and periphery regions.

    Parameters
    ----------
    **data_path** : *(pathlib.PosixPath)* relative path to the segmented masks.

    **method**: *(str)* function which should be applied in order to postprocess masks. Thus far we only use 'split_tumor_into_core_and_periphery' function.

    **method_parameters**: *(Dict)* parameters to use for the function applied in **method** argument.


    Returns
    ------

    output_directory *(pathlib.PosixPath)*: Path where the results have been stored on the disk to.


    Example Usage
    --------------

    ```python

    >>>from src.segmentation import postprocess_masks
    >>>postprocess_masks(
        input_directory = Path('ppdm/data/5IT_DUMMY_STUDY/results/segmentation/tumor/segment___thresholding___method-th_triangle'),
        method = 'split_tumor_into_core_and_periphery',
        method_parameters = {'periphery_as_ratio_of_max_distance': 0.2}
    ```



    """
    module_results_path = "results/segmentation_postprocessing"
    module_name = "postprocess_masks"
    segmentation_postprocessing_function = getattr(
        sys.modules[__name__], method
    )

    parameters_as_string = join_keys_and_values_to_list(method_parameters)
    output_folder_name = join_to_string(
        [module_name, method, *parameters_as_string, data_path.stem]
    )
    output_directory = create_relative_path(
        data_path,
        module_results_path,
        output_folder_name,
        _infer_root_based_on="results",
    )

    directory_ok = check_content_of_two_directories(
        data_path, output_directory
    )

    if directory_ok is False:
        if output_directory.exists():
            remove_content(output_directory)
        segmentation_postprocessing_function(
            data_path, output_directory, **method_parameters
        )
    return output_directory


def _load_and_transform_image(img_path: pathlib.PosixPath) -> dict:
    """
    Load and Downscale image.
    """
    img = _crop_border_from_mask_and_fill_holes(img_path)
    img = custom_rescale(img.astype(np.uint8), 1 / 10)
    return {"name": img_path.name, "image": img}


def _encode_combine_transform_and_save_mask(
    image_info: dict, output_directory: pathlib.PosixPath
) -> None:
    """
    Upscale and save image.
    """
    original_mask = np.load(image_info["image_path"])
    img1 = image_info["core_periphery"][0]
    img2 = image_info["core_periphery"][1]
    img1 = cv2.resize(
        img1, original_mask.shape[::-1], interpolation=cv2.INTER_NEAREST
    )
    img2 = cv2.resize(
        img2, original_mask.shape[::-1], interpolation=cv2.INTER_NEAREST
    )
    img1 = img1 * original_mask
    img2 = img2 * original_mask
    img1[img1 > 0] = 1
    img2[img2 > 0] = 1
    img2 = img2 * 2
    # encode into one image
    img = img1 + img2
    img = img.astype(np.uint8)
    layer_name = image_info["name"]
    image_save_path = pathlib.Path(output_directory) / layer_name
    image_save_path.parent.mkdir(exist_ok=True, parents=True)
    np.save(image_save_path, img)


def split_tumor_into_core_and_periphery(
    input_directory: pathlib.PosixPath,
    output_directory: pathlib.PosixPath,
    periphery_as_ratio_of_max_distance: float = 0.2,
) -> None:

    """
    Function which splits tumor into core and periphery based on relative distance from the surface.

    Parameters
    ----------
    **input_directory** : *(pathlib.PosixPath)* relative path to the segmented masks.

    **output_directory** : *(pathlib.PosixPath)* relative path where the results should be saved to.

    **periphery_as_ratio_of_max_distance**: *(float)* what ratio of the tumor should be consired core. Any number in <0;1>

    Returns
    ------

    None: Results are stored on the disk.


    Example Usage
    --------------

    ```python

    >>>from src.segmentation import split_tumor_into_core_and_periphery
    >>>split_tumor_into_core_and_periphery(
        input_directory = Path('ppdm/data/5IT_DUMMY_STUDY/results/segmentation/tumor/segment___thresholding___method-th_triangle'),
        output_directory = Path('ppdm/data/my_tumor_folder'),
        periphery_as_ratio_of_max_distance = 0.5)
    ```



    """

    original_images = sorted(list(input_directory.glob("*.npy")))
    imgs_tumor_mask_list = parallel(
        _load_and_transform_image,
        original_images,
        n_workers=12,
        threadpool=True,
        progress=True,
    )
    # check if layers (baesed on image names) are ordered correctly
    layer_names = [x["name"] for x in imgs_tumor_mask_list]
    imgs_tum = _stack_and_downscale_images(imgs_tumor_mask_list)
    # do distance transform
    imgs_tum_dt = edt.edt(imgs_tum, black_border=False, parallel=12, order="C")
    # split masked object into core and periphery based on distance from surface
    distance_threshold = periphery_as_ratio_of_max_distance * np.max(
        imgs_tum_dt
    )
    imgs_outer = np.where(imgs_tum_dt < distance_threshold, imgs_tum_dt, 0,)

    # inner mask
    # make sure inner mask is binary
    outer_mask = np.where(imgs_outer > 0, 1, 0)
    # outer mask
    inner_mask = imgs_tum - outer_mask
    # resize z-axis back to original
    inner_mask = zoom(inner_mask, (10, 1, 1), mode="nearest", order=0)
    outer_mask = zoom(outer_mask, (10, 1, 1), mode="nearest", order=0)
    # save to disk
    mask_path = output_directory
    mask_path.parent.mkdir(exist_ok=True, parents=True)
    inner_mask_iterable = [x for x in inner_mask]
    outer_mask_iterable = [x for x in outer_mask]
    inner_outer_mask_individual_layers = [
        {
            "name": layer_name,
            "image_path": layer_image,
            "core_periphery": core_periphery,
        }
        for layer_name, layer_image, core_periphery in zip(
            layer_names,
            original_images,
            zip(inner_mask_iterable, outer_mask_iterable),
        )
    ]
    parallel(
        partial(
            _encode_combine_transform_and_save_mask, output_directory=mask_path
        ),
        inner_outer_mask_individual_layers,
        n_workers=12,
        progress=True,
        threadpool=True,
    )


def _stack_and_downscale_images(
    imgs_tumor_mask_list: List[np.ndarray],
) -> np.ndarray:
    """
    Donwscale and stack 2d images into 3d.
    """
    # now get only images to list
    imgs_tumor_mask = [x["image"] for x in imgs_tumor_mask_list]
    # stack images to create 3d object
    imgs_tum = np.stack(imgs_tumor_mask, axis=0)
    # make sure object mask is binary
    imgs_tum[imgs_tum > 0] = 1
    # resize also z axis
    imgs_tum = zoom(imgs_tum, (0.1, 1, 1), mode="nearest", order=0)
    return imgs_tum


def _crop_border_from_mask_and_fill_holes(
    img_mask_path: pathlib.PosixPath,
    output_directory: pathlib.PosixPath = None,
    pixel_border: int = 0,
) -> None:
    """
    This function removes border pixels of the tumor and
    save them as a np array to the same
    folder where input data are located.
    """

    # Load The Mask
    img_mask = np.load(img_mask_path)

    # Save the original size of the image
    orig_size = img_mask.shape

    # Label The mask
    labeled = label(img_mask)

    # Calculate the counts for each pixels
    values = list(Counter(labeled.ravel()).values())

    # The most common value apart from zero
    idx = np.argmax(values[1:]) + 1

    # Get the entire Image background
    background = labeled == idx

    # Invert background
    background_inverted = ~background

    # Label the background and foreground of the tumor
    foreground = label(background_inverted)
    # Label The background eyt again
    foreground = foreground == 1

    # Rescale down the image so that erosion is performed quickly
    foreground = custom_rescale(foreground.astype(np.uint8), 0.1)

    # Erode the tumor to remove the borders
    eroded_img = dilation(foreground, disk(pixel_border))

    # Rescale it back to the original size
    eroded_img = cv2.resize(
        eroded_img, orig_size[::-1], interpolation=cv2.INTER_NEAREST
    )

    eroded_img = ~eroded_img.astype(bool)
    eroded_img = eroded_img * 1
    eroded_img = eroded_img.astype(np.uint8)
    if output_directory is not None:
        file = output_directory / img_mask_path.name
        file.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(file), eroded_img)
        return None
    return eroded_img


def _parallel_crop_border_from_mask_and_fill_holes(
    data_path: pathlib.PosixPath, **kwargs
):
    """
    This function calculates median "time-series" z-axis threshold value.
    """
    partial_func = partial(
        _crop_border_from_mask_and_fill_holes,
        pixel_border=kwargs["pixel_border"],
        output_directory=kwargs["output_directory"],
    )

    parallel(
        partial_func,
        sorted(list(data_path.glob("*.npy"))),
        n_workers=12,
        progress=True,
        threadpool=True,
    )
