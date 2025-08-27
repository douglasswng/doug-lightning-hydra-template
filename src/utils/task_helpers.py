import warnings
from collections.abc import Callable
from typing import Any

import torch
from omegaconf import DictConfig

from utils.ranked_logger import RankedLogger

log = RankedLogger(__name__)


def process_extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings

    :param cfg: A DictConfig object containing the config tree.
    """
    extras = cfg.get("extras")
    if not extras:
        log.info("No extras config provided, skipping optional utilities")
        return

    assert isinstance(extras, DictConfig), "Extras config must be a DictConfig!"

    if extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")


type TaskFunc = Callable[[DictConfig], tuple[dict[str, Any], dict[str, Any]]]


def exception_wrapper(task_func: TaskFunc) -> TaskFunc:
    """Optional decorator that controls the failure behavior when executing the task function.

    :param task_func: The task function to be wrapped.

    :return: The wrapped task function.
    """

    def wrap(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            metric_dict, object_dict = task_func(cfg)
        except Exception as ex:
            log.exception("")
            raise ex
        finally:
            log.info(f"Output dir: {cfg.paths.output_dir}")
        return metric_dict, object_dict

    return wrap


def get_metric_value(metric_dict: dict[str, Any], metric_name: str) -> float:
    """Safely retrieves value of the metric logged in LightningModule.

    :param metric_dict: A dict containing metric values.
    :param metric_name: If provided, the name of the metric to retrieve.
    :return: If a metric name was provided, the value of the metric.
    """
    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict.get(metric_name)

    assert isinstance(metric_value, torch.Tensor), "Metric value must be a tensor!"

    metric_value = metric_value.item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value
