"""
Module flow logging using Mlflow library.
"""
# pylint: disable=R0913, R0914

import json
import os
import subprocess
from pathlib import Path
from typing import List, Union

import mlflow
import numpy as np

from src.profiles import vizualize_profile
from src.utils import flatten, pathlib_to_string


def mlflow_logging(
    experiment_data: dict,
    profiles: dict,
    mlflow_tracking_uri: List[Union[str, Path]],
    mlflow_experiment_name: str,
    run_name: str,
    script_name: str,
) -> None:
    """
    Function to log experiment artifacts using Mlflow library.
    The entire experiment is saved and the results can be inspected in mlflow dashboard.

    Args:
        experiment_data (dict): [description]
        profiles (dict): [description]
        mlflow_tracking_uri (List[Union[str, Path]]): [description]
        mlflow_experiment_name (str): [description]
        run_name (str): [description]
        script_name (str): [description]
    """
    # mlflow settings
    mlflow.set_tracking_uri(str(mlflow_tracking_uri))
    mlflow.set_experiment(mlflow_experiment_name)
    run = mlflow.start_run(run_name=run_name)

    with run:
        # log experiment script as it was runned
        script_path = (
            Path(os.path.dirname(os.path.realpath("__file__"))) / script_name
        )
        mlflow.log_text(script_path.read_text(), "script.py")
        mlflow.set_tags(
            {
                "mlflow.source.name": str(script_path),
                "mlflow.source.git.commit": get_git_commit_hash(),
                "mlflow.source.git.branch": get_git_branch(),
                "mlflow.source.git.repoURL": get_git_repo_url(),
            }
        )
        # log experiment data
        experiment_json_path = (
            Path(mlflow_tracking_uri).parent / "experiment.json"
        )
        with open(experiment_json_path, "w", encoding="utf-8") as file:
            json.dump(
                pathlib_to_string(experiment_data),
                file,
                ensure_ascii=False,
                indent=4,
            )
        mlflow.log_artifact(experiment_json_path)
        os.remove(experiment_json_path)
        # log experiment data as parameters
        mlflow.log_params(flatten(pathlib_to_string(experiment_data)))
        # log profiles
        for profile in profiles:
            profile_df = profiles[profile]
            profile_fig = vizualize_profile(profile_df, plot=False)
            # data
            profile_csv_path = (
                Path(mlflow_tracking_uri).parent / f"{profile}.csv"
            )
            profile_df.to_csv(profile_csv_path, index=True)
            mlflow.log_artifact(profile_csv_path, "profile_data")
            os.remove(profile_csv_path)
            # metric
            for index, row in profile_df.iterrows():
                if np.isnan(row["virus"]):
                    print(index)
                    continue
                mlflow.log_metric(profile, row["virus"], int(row["distance"]))
            # figure
            mlflow.log_figure(profile_fig, f"profile_figures/{profile}.png")
    print(f"Done, run_id: {run.info.run_id}")


def get_git_commit_hash():
    """
    Get current commit hash.
    """
    return (
        subprocess.check_output(["git", "rev-parse", "HEAD"])
        .decode("utf-8")
        .replace("\n", "")
    )


def get_git_branch():
    """
    Get current branch.
    """
    return (
        subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        .decode("utf-8")
        .replace("\n", "")
    )


def get_git_repo_url():
    """
    Get current repo url.
    """
    return (
        subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"]
        )
        .decode("utf-8")
        .replace("\n", "")
    )
