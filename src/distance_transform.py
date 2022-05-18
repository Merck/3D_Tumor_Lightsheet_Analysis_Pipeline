"""
## distance transform module:
### module for computing distance transform on segmented masks
#### This module consists of one main functions:

* calculate_distance_tranform
"""

# pylint: disable=W0105, C0103, R0914, W0612, W0105, C0301

import pathlib
from pathlib import Path

import edt
import numpy as np
from tqdm import tqdm

from src.utils import (
    check_content_of_two_directories,
    create_relative_path,
    join_to_string,
    log_step,
    remove_content,
)


@log_step
def calculate_distance_tranform(
    data_path: pathlib.PosixPath, stack_size: int = 100
) -> pathlib.PosixPath:

    """
    Wrapper function that calculates the distance transform on the segmented blood vessels masks. Due to the size of the data it calculated the distance transform using overlapping cubes.

    Parameters
    ----------

    **data_path**: *(pathlib.PosixPath)* relative path for the segmented masks to be used for the distance transform

    **stack_size**: *(int)* For the HPC setting we used stack size of 100. For local PC we recommend smaller stack size. [HERE](Code use pre-requisites and Installation instructions.md)

    Returns
    ------
    **output_directory**: *(pathlib.PosixPath) outputs relative path for the folders with calculated distance transform.

    Results are stored on the disk.

    Example Usage
    --------------
    ```python

    >>>from src.distance_tranform import calculate_distance_tranform
    >>>segmented_vessels_path = Path('ppdm/data/5IT_DUMMY_STUDY/results/segmentation/vessel/segment___unet___model_file-model')

    >>>calculate_distance_tranform(study_paths)

    >>>output:
    Path('ppdm/data/5IT_DUMMY_STUDY/results/distance_transform/vessel/distance_tranform___segment___unet___model_file-model')

    ```
    """

    module_results_path = "results/distance_transform"
    module_name = "distance_tranform"
    output_folder_name = join_to_string([module_name, data_path.stem])

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
        # step1 - create overlapping bricks
        submodule_name = "tmp_overlapping_bricks"
        output_folder_name = join_to_string([submodule_name, data_path.stem])

        output_directory = create_relative_path(
            data_path,
            module_results_path,
            output_folder_name,
            _infer_root_based_on="results",
        )
        if output_directory.exists():
            remove_content(output_directory)

        overlapping_bricks = _merge_with_overlap(
            data_path, output_directory, stack_size=stack_size
        )
        # step2 - calculate distance transform on overlapping brikcs
        submodule_name = "tmp_dt_overlapping_bricks"
        output_folder_name = join_to_string([submodule_name, data_path.stem])

        output_directory = create_relative_path(
            data_path,
            module_results_path,
            output_folder_name,
            _infer_root_based_on="results",
        )
        if output_directory.exists():
            remove_content(output_directory)

        overlapping_bricks_dt = _compute_distance_transform(
            overlapping_bricks, output_directory
        )
        # step3 - aggregate information from overlapping bricks
        submodule_name = "tmp_dt_aggregated_bricks"
        output_folder_name = join_to_string([submodule_name, data_path.stem])

        output_directory = create_relative_path(
            data_path,
            module_results_path,
            output_folder_name,
            _infer_root_based_on="results",
        )
        if output_directory.exists():
            remove_content(output_directory)

        aggregated_bricks_dt = _aggregate(
            overlapping_bricks_dt, output_directory
        )
        # final step4 - slice bricks back into individual layers
        output_folder_name = join_to_string([module_name, data_path.stem])

        output_directory = create_relative_path(
            data_path,
            module_results_path,
            output_folder_name,
            _infer_root_based_on="results",
        )
        if output_directory.exists():
            remove_content(output_directory)
        names_of_individual_layers = data_path
        _split(
            aggregated_bricks_dt, output_directory, names_of_individual_layers
        )
    return output_directory


@log_step
def _merge_with_overlap(
    input_paths: pathlib.PosixPath,
    save_path: pathlib.PosixPath,
    stack_size: int,
) -> None:
    """
    Creates overlapping numpy arrays and saves them to the disk.

    Args:
        input_paths (pathlib.PosixPath)
        save_path (pathlib.PosixPath)
        stack_size (int)

    Returns:
        save_path (pathlib.PosixPath)
    """
    input_paths = sorted(input_paths.glob("*.npy"))
    assert len(input_paths) != 0
    save_path = Path(save_path)

    step = stack_size // 2
    tile_split_points = np.array((range(0, len(input_paths) + 1, step)))
    tiles_coords = list(zip(tile_split_points, tile_split_points[2:]))
    if len(input_paths) % stack_size != 0:
        tiles_coords[-1] = (tiles_coords[-1][0], len(input_paths))

    for c in tqdm(tiles_coords):
        tiles_list = []
        for i in range(c[0], c[1]):
            tile = np.load(input_paths[i])
            tiles_list.append(tile)
        tiles = np.stack(tiles_list, axis=0)
        tiles_save_path = save_path / f"tile_{c[0]:04}_{c[1]:04}.npy"
        tiles_save_path.parent.mkdir(exist_ok=True, parents=True)
        np.save(tiles_save_path, tiles)
    return save_path


@log_step
def _compute_distance_transform(
    data_path: pathlib.PosixPath, ouput_path: pathlib.PosixPath
) -> pathlib.PosixPath:
    """
    Computes distance tranform on the overlapping numpy stacks

    Args:
        overlapping_binary_masks (pathlib.PosixPath)
        ouput_path (pathlib.PosixPath)

    Returns:
        ouput_path (pathlib.PosixPath)
    """
    overlapping_binary_masks = sorted(list(data_path.glob("*")))
    for brick in tqdm(overlapping_binary_masks):
        image = np.load(brick)
        brick_segmentation = image
        brick_segmentation = ~brick_segmentation.astype(bool)
        brick_segmentation = brick_segmentation * 1
        brick_dt = edt.edt(brick_segmentation, black_border=False, parallel=12)
        # save
        dt_path = ouput_path / brick.name
        dt_path.parent.mkdir(exist_ok=True, parents=True)
        np.save(dt_path, brick_dt)
    return dt_path.parent


@log_step
def _aggregate(input_path: pathlib.PosixPath, save_path: pathlib.PosixPath):
    """
    un-overlaps the overlapping layes and saves them to disk as numpy arrays

    Args:
        input_path (pathlib.PosixPath)
        save_path (pathlib.PosixPath)

    Returns:
        save_path (pathlib.PosixPath)
    """
    input_path_list = sorted(list(input_path.glob("*")))

    bricks_coords = [
        np.array(x.stem.split("_")[1:3]).astype(int) for x in input_path_list
    ]
    non_overlapping_bricks = [
        np.array(x.stem.split("_")[1:3]).astype(int)
        for x in input_path_list[::2]
    ]
    non_overlapping_bricks_files = [x.name for x in input_path_list[::2]]

    stack_size = bricks_coords[0][1] - bricks_coords[0][0]
    # check 4 equal parts possible

    assert stack_size % 4 == 0

    q_25 = stack_size // 4
    q_50 = stack_size // 2
    q_75 = stack_size - q_25

    for idx, batch in enumerate(non_overlapping_bricks_files):
        # brick to merge
        # merge from these bricks
        which_bricks = [
            x
            for x in bricks_coords
            if non_overlapping_bricks[idx][0] + 1 in list(range(x[0], x[1]))
            or non_overlapping_bricks[idx][1] - 1 in list(range(x[0], x[1]))
        ]
        # load 1 by 1 and stack
        brick_list = []
        for j, which_brick in enumerate(which_bricks):
            file_path = (
                input_path
                / f"tile_{which_brick[0]:04}_{which_brick[1]:04}.npy"
            )
            # start or end
            if len(which_bricks) == 2:
                # start
                if idx == 0:
                    if j == 0:
                        brick = np.load(file_path)
                        brick = brick[:q_75]
                        brick_list.append(brick)
                    else:
                        brick = np.load(file_path)
                        brick = brick[q_25:q_50]
                        brick_list.append(brick)
                # end
                if idx == len(non_overlapping_bricks_files) - 1:
                    if j == 0:
                        brick = np.load(file_path)
                        brick = brick[q_50:q_75]
                        brick_list.append(brick)
                    else:
                        brick = np.load(file_path)
                        brick = brick[q_25:]
                        brick_list.append(brick)
            # elswhere
            else:
                if j == 0:
                    brick = np.load(file_path)
                    brick = brick[q_50:q_75]
                    brick_list.append(brick)
                elif j == 1:
                    brick = np.load(file_path)
                    brick = brick[q_25:q_75]
                    brick_list.append(brick)
                elif j == 2:
                    brick = np.load(file_path)
                    brick = brick[q_25:q_50]
                    brick_list.append(brick)
            # stack the partial bricks
            brick = np.vstack(brick_list)
            # save it
            save_b = save_path / batch
            save_b.parent.mkdir(exist_ok=True, parents=True)
            np.save(save_b, brick)
    return save_path


@log_step
def _split(
    overlapping_layers_path: pathlib.PosixPath,
    save_path: pathlib.PosixPath,
    individual_layer_names: pathlib.PosixPath,
) -> None:
    """
    cut stacked numpy arrays into individual z-planes arrays and saves them to disk.

    Args:
        overlapping_layers_path (pathlib.PosixPath)
        save_path (pathlib.PosixPath)
        segmented_vessels (pathlib.PosixPath)

    Returns:
        save_path (pathlib.PosixPath)
    """
    single_layer_path_list = [
        i.stem for i in sorted(list(individual_layer_names.glob("*")))
    ]
    for brick_path in tqdm(sorted(list(overlapping_layers_path.glob("*")))):
        brick = np.load(brick_path)
        start, end = np.array(brick_path.stem.split("_")[1:]).astype(int)

        for i in range(brick.shape[0]):
            layer = brick[i]
            layer_index = start + i
            layer_name = Path(single_layer_path_list[layer_index]).name
            layer_save_path = save_path / layer_name
            layer_save_path.parent.mkdir(exist_ok=True, parents=True)
            np.save(layer_save_path, layer)
    return save_path
