"""
## segmentation module:
### module for blood vessels and tumor segmentation
#### This module consists of four main functions:

* random_forest
* segmentation_wrapper
* thresholding
* unet

"""

# pylint: disable=C0301, E1121

import pathlib
import sys
from functools import partial
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from fastcore.parallel import parallel
from joblib import load
from PIL import Image
from skimage import feature, future
from skimage.filters import threshold_otsu, threshold_triangle, threshold_yen
from torchvision import transforms
from tqdm import tqdm

from src.utils import (
    check_content_of_two_directories,
    create_relative_path,
    join_keys_and_values_to_list,
    join_to_string,
    log_step,
    remove_content,
    select_avail_gpu,
)
from src.volume_slicer import VolumeSlicer


@log_step
def segmentation_wrapper(
    data_path: pathlib.PosixPath, method: str, method_parameters: Dict
) -> pathlib.PosixPath:

    """

    This function is a main wrapper for segmentation used within the automated pipeline.

    It uses functions written in this script and wraps them in this wrapper (one big function).
    If one want to use segmentations methods in a script, you may use individual segmentation functions -> unet, thresholding, random_forest


    Parameters
    ----------

    **data_path** : *(pathlib.PosixPath object)*
    relative path to the study which should be segmented (notice that we work with preprocessed images which are in npy for and not in .tiff format)

    **method** : *(str)*
    method to use for segmentation (unet, thresholding, random_forest)

    **method_parameters**: *(Dict)*
    relative path to the pre-trained model binary file, which should be used for the segmentation

    **device**: *(str)*
    graphic card which should be used. Code automatically selects graphic card, that is free.

    Returns
    ------

    None: Results are dumbed on the disk.


    Example Usuage
    --------------

    ```python

    >>>from src.segmentation import unet
    >>>unet(
        input_directory = Path('ppdm/data/5IT_DUMMY_STUDY/source/transformed/np_and_resized/vessel'),
        output_directory = Path('ppdm/data/5IT_DUMMY_STUDY/DOCUMENTATION/raw/vessel'),
        model_file = Path('pretrained_models/unet_train_full_5IT_7IV_8IV.pt')
        )
    ```

    """

    module_results_path = "results/segmentation"
    module_name = "segment"
    segmentation_function = _select_method(method)
    parameters_as_string = join_keys_and_values_to_list(method_parameters)
    output_folder_name = join_to_string(
        [module_name, method, *parameters_as_string]
    )

    output_directory = create_relative_path(
        data_path, module_results_path, output_folder_name
    )

    directory_ok = check_content_of_two_directories(
        data_path, output_directory
    )

    if directory_ok is False:
        if output_directory.exists():
            remove_content(output_directory)
        segmentation_function(data_path, output_directory, **method_parameters)
    return output_directory


@log_step
def unet(
    input_directory: pathlib.PosixPath,
    output_directory: pathlib.PosixPath,
    model_file: pathlib.PosixPath,
    device: str = None,
) -> None:
    """
    Function that performs the segmentation of the blood vessels (if you want to segment e.g. tumors this function can still be used. However,
    you need to provide a pre-trained model (model_file) that is pretrained for tumors)

    Function performs the segmentation and saves the results (individual segmented images) to the output_directory path provided.


    Parameters
    ----------

    **input_directory** : *(pathlib.PosixPath object)*
    relative path to the images which should be segmented (notice that we work with preprocessed images which are in npy for and not in .tiff format)

    **output_directory** : *(pathlib.PosixPath object)*
    relative path to a folder where the results should be saved

    **model_file**: *(pathlib.PosixPath object)*
    relative path to the pre-trained model binary file, which should be used for the segmentation

    **device**: *(str)*
    graphic card which should be used. Code automatically selects graphic card, that is free.

    Returns
    ------

    None: Results are dumbed on the disk.


    Example Usuage
    --------------

    ```python

    >>>from src.segmentation import unet
    >>>unet(
        input_directory = Path('msd_projects/ppdm/data/5IT_DUMMY_STUDY/source/transformed/np_and_resized/vessel'),
        output_directory = Path('msd_projects/ppdm/data/5IT_DUMMY_STUDY/DOCUMENTATION/raw/vessel'),
        model_file = Path('pretrained_models/unet_train_full_5IT_7IV_8IV.pt')
        )
    ```
    """
    input_directory_paths = sorted(list(input_directory.glob("*.npy")))

    directory_ok = check_content_of_two_directories(
        input_directory, output_directory
    )

    if directory_ok is False:
        if output_directory.exists():
            remove_content(output_directory)

        if device is None:
            device = f"cuda:{select_avail_gpu()}"
            model = torch.load(model_file)
            model = model.to(device).eval()

        for file in tqdm(input_directory_paths):
            image = np.load(file)
            # convert 16 bit images to 8bits
            if image.dtype == "uint16":
                image = image / 65535 * 255
                image = image.astype(np.uint8)
            image = np.expand_dims(image, axis=0)
            # split images to tiles
            tiler = VolumeSlicer(
                image.shape,
                voxel_size=(image.shape[0], 512, 512),
                voxel_step=(image.shape[0], 512, 512),
            )
            tiles = tiler.split(image)
            tiles_processed = _unet_runner(tiles, model, device)
            # merge tiles back to one image
            tiles_stiched = tiler.merge(tiles_processed)
            mask = tiles_stiched[0, :, :]
            # save mask
            save_path_mask = output_directory / file.name
            output_directory.mkdir(parents=True, exist_ok=True)
            np.save(save_path_mask, mask.astype(np.uint8))


def _batch(iterable: Union[List, Tuple], batch_size: int = 1) -> List[List]:
    """
    Split iterable by batch_size.
    """
    iterable_len = len(iterable)
    for ndx in range(0, iterable_len, batch_size):
        yield iterable[ndx : min(ndx + batch_size, iterable_len)]


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    "Convert pytorch tensor to numpy array."
    return tensor.detach().cpu().numpy()


def _unet_runner(
    image_list: List[np.ndarray], model: pathlib.PosixPath, device: str
) -> List[np.ndarray]:
    """
    Perform inference for a list of images.
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                # Pytorch inference: Values of Imagenet Dataset for normalizing (Transfer Learning)
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    segmented_results = []
    for i in _batch(image_list, batch_size=10):
        img_transformed = [
            transform(Image.fromarray(x[0]).convert("RGB")).unsqueeze(0)
            for x in i
        ]

        img_transformed = torch.cat(img_transformed).to(device)
        model.eval()
        with torch.no_grad():
            predictions_list = model(img_transformed)

        for prediction in predictions_list:
            prediction = prediction.unsqueeze(0)
            prediction = _to_numpy(prediction)
            prediction = np.where(
                prediction[:, 0, :, :] > prediction[:, 1, :, :], 0, 1
            )
            segmented_results.append(prediction)
    return segmented_results


@log_step
def thresholding(
    data_path: pathlib.PosixPath,
    output_directory: pathlib.PosixPath,
    method: str,
    mask_path: pathlib.PosixPath = None,
) -> None:

    """
    Function that performs the segmentation using thresholding. This function can be used for any channel
    (e.g. vessels, tumors) as it calculates "optimal" threshold value automatically.

    Function performs the segmentation and saves the results (individual segmented images) to the output_directory path provided.


    Parameters
    ----------

    **data_path**: *(pathlib.PosixPath object)*
    relative path to the images which should be segmented (notice that we work with preprocessed images which are in npy for and not in .tiff format)

    **output_directory** *(pathlib.PosixPath object)*
    relative path to a folder where the results should be saved

    **method**: *(str)* segmentation method to be used.
    Supported methods are "th_otsu", "th_triangle", "th_yen".


    ANOTHER OPTIONAL PARAMETERS:

    **mask_path**: *(pathlib.PosixPath object)*
    relative path for the binary mask to be used to filter the area of interest on which the segmentation will be performed.



    Returns
    ------

    None: Results are dumbed on the disk.


    Example Usuage
    --------------
    ```python

    >>>from src.segmentation import thresholding
    >>>thresholding(
        data_path = Path('ppdm/data/5IT_DUMMY_STUDY/source/transformed/np_and_resized/vessel'),
        output_directory = Path('msd_projects/ppdm/data/5IT_DUMMY_STUDY/THRES/raw/vessel'),
        method = "th_otsu"
    )

    ```

    """

    supported_methods = ["th_otsu", "th_triangle", "th_yen"]
    if method not in supported_methods:
        raise ValueError(
            f"thresholding method must be either in : {supported_methods}, you have provided: {method}"
        )

    threshold_value = _calculate_threshold_value(data_path, method, mask_path)
    print(f"threshold {method} value: {threshold_value}")
    _segment_and_save_masks_threshold(
        data_path, threshold_value, output_directory
    )


@log_step
def random_forest(
    data_path: pathlib.PosixPath,
    output_directory: pathlib.PosixPath,
    model_file,
) -> None:

    """
    Function that performs the segmentation of the blood vessels using Random Forest model (if you want to segment e.g. tumors this function can still be used. However,
    you need to provide a pre-trained model (model_file) that is pretrained for tumors)

    Function performs the segmentation and saves the results (individual segmented images) to the output_directory path provided.


    Parameters
    ----------

    **data_path**: *(pathlib.PosixPath object)*
    relative path to the images which should be segmented (notice that we work with preprocessed images which are in npy for and not in .tiff format)

    **output_directory** *(pathlib.PosixPath object)*
    relative path to a folder where the results should be saved

    **model_file**: *(pathlib.PosixPath object)*
    relative path to the pre-trained model binary file, which should be used for the segmentation


    Returns
    ------

    None: Results are dumbed on the disk.


    Example Usuage
    --------------
    ```python

    >>>from src.segmentation import random_forest
    >>>random_forest(
        data_path = Path('ppdm/data/5IT_DUMMY_STUDY/source/transformed/np_and_resized/vessel'),
        output_directory = Path('ppdm/data/5IT_DUMMY_STUDY/RF/raw/vessel'),
        model_file = Path('pretrained_models/rf_model.joblib')
    )
    ```
    """

    loaded_rf = load(model_file)

    segment_rf_partial = partial(
        _segment_random_forest,
        loaded_estimator_to_use=loaded_rf,
        output_directory=output_directory,
    )

    directory_ok = check_content_of_two_directories(
        data_path, output_directory
    )

    if directory_ok is False:
        if output_directory.exists():
            remove_content(output_directory)

        parallel(
            segment_rf_partial,
            sorted(list(data_path.glob("*.npy"))),
            n_workers=12,
            progress=True,
            threadpool=True,
        )


def _calculate_threshold_value(data_path, method, mask_path=None) -> float:
    """
    This function calculates median "time-series" z-axis threshold value.
    """
    partail_func = partial(
        _parallel_calculate_threshold_values,
        method=method,
        mask_path=mask_path,
    )
    paralel_hist_counts = parallel(
        partail_func,
        sorted(list(data_path.glob("*.npy"))),
        n_workers=12,
        progress=True,
        threadpool=True,
    )
    return pd.DataFrame(paralel_hist_counts)["threshold"].median()


def _parallel_calculate_threshold_values(
    img_path: pathlib.PosixPath,
    mask_path: pathlib.PosixPath,
    method: str = None,
) -> dict:
    """
    Load image (stored numpy array) and calculate Otsu and Triangle threshold for one uint16 image.

    Args:
        one_img (np.array): uint16 numpy array (original size)
    """
    img_i = np.load(img_path)
    if mask_path is not None:
        mask_i = np.load(mask_path)
        img_i = img_i * mask_i

    if method == "th_otsu":
        threshold_value = threshold_otsu(img_i)

    elif method == "th_triangle":
        threshold_value = threshold_triangle(img_i)

    elif method == "th_yen":
        threshold_value = threshold_yen(img_i)

    return {"threshold": threshold_value}


def _segment_and_save_masks_threshold(
    data_path: pathlib.PosixPath,
    threshold_value: float,
    output_directory: pathlib.PosixPath,
) -> None:
    """
    This function segments images based on the given threshold value and saves
    masks as a numpy arrays into output directory.

    """

    img_list = sorted(list(data_path.glob("*.npy")))

    directory_ok = check_content_of_two_directories(
        data_path, output_directory
    )

    if directory_ok is False:
        if output_directory.exists():
            remove_content(output_directory)

        for image_i_name in tqdm(img_list):
            image_i = np.load(image_i_name)
            image_i_mask = np.where(image_i > threshold_value, 1, 0)
            final_file = output_directory / image_i_name.name
            output_directory.mkdir(exist_ok=True, parents=True)
            final_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(final_file, image_i_mask.astype(np.uint8))


features_fun = partial(
    feature.multiscale_basic_features,
    intensity=True,
    edges=True,
    texture=True,
    sigma_min=1,
    sigma_max=16,
    multichannel=False,
)


def _segment_random_forest(
    img_in_path: pathlib.PosixPath,
    loaded_estimator_to_use,
    output_directory: pathlib.PosixPath,
) -> None:

    """
    This function performs segmentation by RF model.
    Acceps one Image so that parallel computing can be uttilized

    Args:
        img_in_path (pathlib): pathlib to the np image to be segmented
        estimator_to_use : estimator model to be used for the segmentation
        dest (pathlib): destination folder where the segmented images should be saved to
    """

    img_in = np.load(img_in_path)

    x_design_matrix = features_fun(img_in)

    n_features = x_design_matrix.shape

    x_design_matrix_reshaped = x_design_matrix.reshape(-1, n_features[2])
    pred = future.predict_segmenter(
        x_design_matrix_reshaped, loaded_estimator_to_use
    )
    pred_reshaped = pred.reshape(n_features[0], n_features[1])
    if np.max(pred_reshaped) == 2:
        pred_reshaped = pred_reshaped - 1

    output_directory.mkdir(parents=True, exist_ok=True)

    final_file = output_directory / img_in_path.name
    np.save(final_file, pred_reshaped.astype(np.uint8))


#     np.save(output_directory / str(img_in_path.stem + ".npy"), pred_reshaped)


def _load_numpy_array_generator(sorted_list: List):
    """
    Generator for loading a list of np arrays
    """
    # Sorting again for good measure
    sorted_list = sorted(sorted_list)

    for element_i in sorted_list:
        yield np.load(element_i)


def _select_method(method) -> str:
    """
    Helper function for selecting attribute from current script
    """
    return getattr(sys.modules[__name__], method)
