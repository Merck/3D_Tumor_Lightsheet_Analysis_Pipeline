"""
## Profiles module:
### Computing and Vizualizing final profile
#### This module consists of two main functions:

* calculate_profile
* vizualize_profile
"""

# pylint: disable=R0914,C0301
import pathlib
import pickle
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastcore.parallel import parallel
from tqdm import tqdm

from src.utils import create_relative_path, log_step, rescale_column_pandas


@log_step
def calculate_profile(
    viral_intensity: pathlib.PosixPath,
    distance_from_blood_vessel: pathlib.PosixPath,
    tumor_mask: pathlib.PosixPath,
    pixel_to_microns: float = 4,
    force_overwrite: bool = True,
) -> Dict:

    """
    High level function that calculates profile of viral intensity binned by distance from blood vessel for tumor area.
    It requires *Virus* channel, *Distance Transform of blood vessels* and *Tumor masks* in order to aggregate the final profile.



    Parameters
    ----------

     **viral_intensity**: *(pathlib.PosixPath)* Relative path to the transformed (resized and numpy format) of Virus channel.

     **distance_from_blood_vessel**: *(pathlib.PosixPath)* Relative path to the distance transform arrays calculated from the segmented blood vessels masks.

     **tumor_mask**: *(pathlib.PosixPath)* Relative path to the preprocessed tumor masks. It can also be multi-color masks for the core-periphery separation.

      **pixel_to_microns** *(float)* Ration between pixels and microns. (When Images multiplied by 4)

      **force_overwrite** *(bool)* if true profiles will be recalculated from scratch even if it has been calculated before.

    Returns
    ------
    **Dict**: *(dict) dictionary containign the results in a nested form. One Key are profiles calcualted for entire tumor, another are calcualted only for core etc .

    Example Usuage
    --------------

    ```python

    >>>from src.profiles import calculate_profile


    >>>viral_intensity = Path('ppdm/data/5IT_DUMMY_STUDY/source/transformed/np_and_resized/virus')
    >>>distance_from_blood_vessel = Path('ppdm/data/5IT_DUMMY_STUDY/results/distance_transform/vessel/distance_tranform___segment___unet___model_file-model')
    >>>tumor_mask = Path('ppdm/data/5IT_DUMMY_STUDY/results/segmentation_postprocessing/tumor/postprocess_masks___split_tumor_into_core_and_periphery___segment___thresholding___method-th_triangle')
    >>>pixel_to_microns = 4
    >>>force_overwrite=False

    >>>profiles = calculate_profile(
    viral_intensity = viral_intensity,
    distance_from_blood_vessel = distance_from_blood_vessel,
    tumor_mask = tumor_mask,
    pixel_to_microns = pixel_to_microns,
    force_overwrite=force_overwrite)


    >>>profiles
    >>>output:

    {'all':        distance       virus
       4.000000  721.423291
       6.648681  721.463944
      10.449533  720.214587
      14.169995  716.011269
      18.046974  715.064906
    ```


    """

    module_results_path = "results/profiles"

    virus_paths = sorted(list(Path(viral_intensity).glob("*")))
    distance_paths = sorted(list(Path(distance_from_blood_vessel).glob("*")))
    tumor_mask_paths = sorted(list(Path(tumor_mask).glob("*")))

    output_directory = create_relative_path(
        viral_intensity,
        module_results_path,
        f"{distance_from_blood_vessel.parts[-1]}/{tumor_mask.parts[-1]}/pixel_to_microns-{pixel_to_microns}",
    )
    output_file_path = output_directory / "profiles.pickle"

    if output_file_path.exists() and force_overwrite is False:
        with open(output_file_path, "rb") as file:
            profiles = pickle.load(file)
    else:
        inputs = [
            {"distance": distance, "virus": virus, "tumor": tumor}
            for distance, virus, tumor in zip(
                distance_paths, virus_paths, tumor_mask_paths
            )
        ]
        profiles_list = parallel(
            _profile_data_from_one_layer_wrapper,
            inputs,
            n_workers=12,
            progress=True,
            threadpool=True,
        )

        profiles = _aggregate_profiles(profiles_list)
        # convert distance to microns
        for key in profiles:
            profiles[key] = rescale_column_pandas(
                profiles[key], "distance", pixel_to_microns
            )
            profiles[key].reset_index(drop=True, inplace=True)
        # save results
        with open(output_file_path, "wb") as file:
            pickle.dump(profiles, file)
    return profiles


def _profile_data_from_one_layer_wrapper(data: dict) -> dict:
    """
    Wrapper that loads data to numpu before calling 'profile_data_from_one_layer'.
    """
    distance = np.load(data["distance"])
    virus = np.load(data["virus"])
    tumor = np.load(data["tumor"])
    return _profile_data_from_one_layer(distance, virus, tumor)


def _profile_data_from_one_layer(
    distance_from_blood_vessel: np.ndarray,
    viral_intensity: np.ndarray,
    tumor_mask: np.ndarray,
) -> Dict:
    """
    For each layer bin viral intensity by distance from blood vessel, but only in tumor area.
    """
    profiles = {}
    cell_coords_dict = {
        "all": np.nonzero(tumor_mask),
        "core": np.nonzero(tumor_mask == 1),
        "periphery": np.nonzero(tumor_mask == 2),
    }
    for key in cell_coords_dict:  # why? do you expect more keys later?
        coords = cell_coords_dict[key]
        distance = distance_from_blood_vessel[coords].ravel()
        virus = viral_intensity[coords].ravel()
        if len(distance) == 0:
            data_frame = pd.DataFrame({"distance": [0], "virus": [0]})
        else:
            data_frame = pd.DataFrame({"distance": distance, "virus": virus})
            data_frame = data_frame.loc[
                lambda d: ~d["distance"].isin([np.inf])
            ]

        ranges = np.arange(
            data_frame.distance.min(), data_frame.distance.max() + 1, 1
        )
        groups = data_frame.groupby(pd.cut(data_frame.distance, ranges))
        profile_sum = groups.sum()
        profile_count = groups.count()
        profile_sum = profile_sum.dropna()
        profile_count = profile_count.dropna()
        profiles[key] = {"sum": profile_sum, "count": profile_count}
    return profiles


def _aggregate_profiles(profiles: List) -> pd.DataFrame:
    """
    Aggregate data from individual layers and calculate average
    viral intensity binned by distance from blood vessel.
    """
    profiles_agg = {
        "all": {"sum": [], "count": []},
        "core": {"sum": [], "count": []},
        "periphery": {"sum": [], "count": []},
    }
    # list of dataframes
    for profile_data in tqdm(profiles):
        for k in profile_data:
            profiles_agg[k]["sum"].append(profile_data[k]["sum"])
            profiles_agg[k]["count"].append(profile_data[k]["count"])
    # concat to one dataframe
    for k in profiles_agg:
        profiles_agg[k]["sum"] = pd.concat(
            profiles_agg[k]["sum"], ignore_index=False, sort=False
        )
        profiles_agg[k]["count"] = pd.concat(
            profiles_agg[k]["count"], ignore_index=False, sort=False
        )
    # group by distance
    for k in profiles_agg:
        # sum
        df_sum = profiles_agg[k]["sum"]
        groups = df_sum.groupby(df_sum.index)
        profiles_agg[k]["sum"] = groups.sum()
        # count
        df_count = profiles_agg[k]["count"]
        groups = df_count.groupby(df_count.index)
        profiles_agg[k]["count"] = groups.sum()
    # calculate average from  sum and count
    for k in profiles_agg:
        profiles_agg[k] = profiles_agg[k]["sum"] / profiles_agg[k]["count"]
    return profiles_agg


def vizualize_profile(
    data_frame: pd.DataFrame,
    distance_threshold_microns: int = 100,
    ylim_bottom: float = None,
    plot: bool = True,
) -> plt.figure:
    """
    Function that vizualizes the aggregated profiles (output of the calculate_profile function)


    Parameters
    ----------

     **data_frame**: *(pd.DataFrame)* Dataframe with aggregated profiles to use for the plot

     **distance_threshold_microns**: *(int)* Maximum distance (microns) on the x-axis.

     **ylim_bottom**: *(float)*  Minumum values to use in the y-axis.

     **plot** *(bool)* Wheater the profile shuld be plotted or not (usefull when plotting multiple profiles in one cell. Otherwise can be ignored)

    Returns
    ------
    None

    Example Usuage
    --------------

    ```python

    >>>from src.profiles import vizualize_profile

    >>>profile_all = vizualize_profile(profiles["all"])
    >>>profile_core = vizualize_profile(profiles["core"])
    >>>profile_periphery = vizualize_profile(profiles["periphery"])
    ```


    """
    profile = data_frame.copy()
    profile_subset = profile[profile.distance <= distance_threshold_microns]
    # log as figure
    fig = plt.figure(figsize=(6, 6))
    plt.rcParams.update({"font.size": 12})
    plt.plot(
        profile_subset["distance"], profile_subset["virus"], color="black"
    )
    plt.xlabel("Distance from blood vessel [microns]", size=14)
    plt.ylabel("Viral Intensity", size=14)
    if ylim_bottom is not None:
        plt.ylim(bottom=ylim_bottom)
    if plot:
        plt.show()
    plt.close()
    return fig
