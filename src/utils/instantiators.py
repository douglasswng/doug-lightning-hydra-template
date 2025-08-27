import hydra
from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from utils.ranked_logger import RankedLogger

logger = RankedLogger(__name__)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> list[Callback]:
    """Instantiates callbacks from config.

    :param callbacks_cfg: A DictConfig object containing callback configurations.
    :return: A list of instantiated callbacks.
    """
    callbacks: list[Callback] = []

    if not callbacks_cfg:
        logger.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for cb_conf in callbacks_cfg.values():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            logger.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_exp_loggers(logger_cfg: DictConfig) -> list[Logger]:
    """Instantiates experiment loggers from config.

    :param logger_cfg: A DictConfig object containing logger configurations.
    :return: A list of instantiated loggers.
    """
    exp_loggers: list[Logger] = []

    if not logger_cfg:
        logger.warning("No logger configs found! Skipping...")
        return exp_loggers

    for lg_conf in logger_cfg.values():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            logger.info(f"Instantiating logger <{lg_conf._target_}>")
            exp_loggers.append(hydra.utils.instantiate(lg_conf))

    return exp_loggers
