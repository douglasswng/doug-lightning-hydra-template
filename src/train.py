from typing import Any

import hydra
import lightning as L
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from utils import (
    RankedLogger,
    exception_wrapper,
    get_metric_value,
    instantiate_callbacks,
    instantiate_exp_loggers,
    log_hyperparameters,
    process_extras,
    save_config,
    save_tags,
)

logger = RankedLogger(__name__)


@exception_wrapper
def train(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    logger.info(f"Saving tags to {cfg.paths.tags_path}")
    save_tags(cfg)

    logger.info(f"Saving config to {cfg.paths.config_path}")
    save_config(cfg)

    logger.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    logger.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    logger.info("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    logger.info("Instantiating experiment loggers...")
    exp_loggers: list[Logger] = instantiate_exp_loggers(cfg.get("logger"))

    logger.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=exp_loggers
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "exp_loggers": exp_loggers,
        "trainer": trainer,
    }

    if exp_loggers:
        logger.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        logger.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        logger.info("Starting testing!")
        if not isinstance(trainer.checkpoint_callback, ModelCheckpoint):
            raise ValueError("Need checkpoint callback to test!")

        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            logger.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        logger.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> float | None:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. disable warnings)
    process_extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_name = cfg.get("optimized_metric")
    if metric_name is None:
        return None

    metric_value = get_metric_value(metric_dict, metric_name)
    return metric_value


if __name__ == "__main__":
    main()
