import subprocess
from pathlib import Path
from typing import List, Union
import numpy as np
from PIL import Image
from pylab import *
import mlflow

from src.metrics import *


def print_auto_logged_info(r):
    tags = {
        k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")
    }
    artifacts = [
        f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")
    ]
    print("run_id: {}".format(r.info.run_id))
    print("artifacts: {}".format(artifacts))
    print("params: {}".format(r.data.params))
    print("metrics: {}".format(r.data.metrics))
    print("tags: {}".format(tags))


def get_git_commit_hash():
    return (
        subprocess.check_output(["git", "rev-parse", "HEAD"])
        .decode("utf-8")
        .replace("\n", "")
    )


def get_git_branch():
    return (
        subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        .decode("utf-8")
        .replace("\n", "")
    )


def get_git_repo_url():
    return (
        subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"]
        )
        .decode("utf-8")
        .replace("\n", "")
    )


def log_metrics(
    y_pred: List[np.ndarray],
    y_true: List[np.ndarray],
    f_names: List[Union[str, Path]],
    layer_nums: List[int],
    mlflow_tracking_uri: List[Union[str, Path]],
    mlflow_experiment_name: str,
    orig_size: tuple = None,
    run_id: str = None,
    run_name: str = "run_name",
):

    assert len(y_pred) == len(y_true) == len(f_names) == len(layer_nums)

    # mlflow settings
    mlflow.set_tracking_uri(str(mlflow_tracking_uri))
    mlflow.set_experiment(mlflow_experiment_name)

    if run_id is not None:
        run = mlflow.start_run(run_id=run_id)
    else:
        run = mlflow.start_run(run_name=run_name)

    with run:
        with mlflow.start_run(run_name="metrics", nested=True):
            # log and add metrics to list
            metrics = {
                "dice": [],
                "precision": [],
                "recall": [],
                "n_positive": [],
                "n_components": [],
                "layer_num": [],
            }

            sort_idxs = np.argsort(layer_nums)

            y_pred = np.array(y_pred)
            y_true = np.array(y_true)
            f_names = np.array(f_names)
            layer_nums = np.array(layer_nums)

            y_pred = y_pred[sort_idxs]
            y_true = y_true[sort_idxs]
            f_names = f_names[sort_idxs]
            layer_nums = layer_nums[sort_idxs]

            predictions = zip(y_pred, y_true)
            for idx, p in enumerate(predictions):
                y_pred = p[0]
                y_true = p[1]

                f = f_names[idx]

                layer_num = layer_nums[idx]

                metrics["layer_num"].append(layer_num)

                cm = confusion_matrix(y_true, y_pred)

                # log dice
                dice_val = dice(y_true, y_pred)
                metrics["dice"].append(dice_val)
                mlflow.log_metric("dice", dice_val, layer_num)
                # log precision
                precision_val = precision(cm)
                metrics["precision"].append(precision_val)
                mlflow.log_metric("precision", precision_val, layer_num)
                # log recall
                recall_val = recall(cm)
                metrics["recall"].append(recall_val)
                mlflow.log_metric("recall", recall_val, layer_num)
                # log n_positive
                n_positive_val = n_positive(y_pred)
                metrics["n_positive"].append(n_positive_val)
                mlflow.log_metric("n_positive", n_positive_val, layer_num)
                # log n_components
                n_components_val = n_components(y_pred)
                metrics["n_components"].append(n_components_val)
                mlflow.log_metric("n_components", n_components_val, layer_num)
                # average all metrics and log them
                mlflow.log_metric("dice_average", np.mean(metrics["dice"]))
                mlflow.log_metric(
                    "precision_average", np.mean(metrics["precision"])
                )
                mlflow.log_metric("recall_average", np.mean(metrics["recall"]))
                mlflow.log_metric(
                    "n_positive_average", np.mean(metrics["n_positive"])
                )
                mlflow.log_metric(
                    "n_components_average", np.mean(metrics["n_components"])
                )

                # save color coded prediction (confusion matrix) as a figure
                color_code_fig = color_code(y_true, y_pred)
                mlflow.log_figure(
                    color_code_fig, f'color_coded_conf_mat/{(f.stem + ".png")}'
                )

                # save prediction resized to original size
                img = y_pred.astype(np.uint8)
                img[img > 0] = 255
                if orig_size is not None:
                    img = Image.fromarray(img).resize(
                        orig_size, resample=Image.NEAREST
                    )
                mlflow.log_image(img, f'mask/{(f.stem + ".png")}')
    return None
