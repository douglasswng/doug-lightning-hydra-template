from pathlib import Path
from typing import Any

import rich
import rich.syntax
import rich.tree
from lightning import LightningModule, Trainer
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf
from rich.console import Console

from utils.ranked_logger import RankedLogger

logger = RankedLogger(__name__)


def _create_config_tree(cfg: DictConfig) -> rich.tree.Tree:
    tree = rich.tree.Tree("CONFIG")
    for field in cfg:
        field_str = str(field)
        branch = tree.add(field_str)
        config_group = cfg[field]

        if isinstance(config_group, DictConfig):
            try:
                branch_content = OmegaConf.to_yaml(config_group, resolve=True)
            except Exception as e:
                # If interpolation fails (e.g., missing hydra.job.num in single runs),
                # fall back to unresolved YAML
                logger.warning(f"Failed to resolve interpolations for {field}: {e}")
                branch_content = OmegaConf.to_yaml(config_group, resolve=False)
        else:
            branch_content = str(config_group)
        branch.add(rich.syntax.Syntax(branch_content, "yaml"))
    return tree


@rank_zero_only
def save_config(cfg: DictConfig) -> None:
    """Prints the config tree to console and saves it to a log file.

    :param cfg: A DictConfig composed by Hydra.
    """
    tree = _create_config_tree(cfg)

    # Print to console
    rich.print(tree)

    # Save to file
    output_file = Path(cfg.paths.config_path)
    with output_file.open("w") as file:
        console = Console(file=file, width=120)
        console.print(tree)


@rank_zero_only
def save_tags(cfg: DictConfig) -> None:
    """Prompts user to input tags from command line if no tags are provided in config.

    :param cfg: A DictConfig composed by Hydra.
    :param save_to_file: Whether to export tags to the hydra output folder. Default is ``False``.
    """
    tags = cfg.get("tags")
    if tags is None:
        raise ValueError("Specify tags")

    logger.info(f"Tags: {tags}")

    tags_path = Path(cfg.paths.tags_path)
    with tags_path.open("w") as file:
        rich.print(cfg.tags, file=file)


@rank_zero_only
def log_hyperparameters(object_dict: dict[str, Any]) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Additionally saves:
        - Number of model parameters (total, trainable, non-trainable)

    :param object_dict: A dictionary containing the following objects:
        - `"cfg"`: A DictConfig object containing the main config.
        - `"model"`: The Lightning model.
        - `"trainer"`: The Lightning trainer.
    """
    # Validate required keys
    required_keys = {"cfg", "model", "trainer"}
    missing_keys = required_keys - object_dict.keys()
    if missing_keys:
        logger.error(f"Missing required keys in object_dict: {missing_keys}")
        return

    trainer: Trainer = object_dict["trainer"]
    if not trainer.loggers:
        logger.warning("No experiment loggers found. Skipping hyperparameter logging.")
        return

    # Convert config to container format
    cfg_container = OmegaConf.to_container(object_dict["cfg"], resolve=True)
    assert isinstance(cfg_container, dict)

    model = object_dict["model"]
    assert isinstance(model, LightningModule)

    # Prepare hyperparameters dictionary
    hparams: dict[str, Any] = {}
    for k, v in cfg_container.items():
        assert isinstance(k, str)
        hparams[k] = v

    # Calculate model parameters efficiently in single pass
    total_params = 0
    trainable_params = 0

    for param in model.parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count

    non_trainable_params = total_params - trainable_params

    # Add model parameter counts
    hparams.update(
        {
            "model/params/total": total_params,
            "model/params/trainable": trainable_params,
            "model/params/non_trainable": non_trainable_params,
        }
    )

    # Log hyperparameters to all available loggers
    logger.info(f"Logging hyperparameters to {len(trainer.loggers)} logger(s)")
    for exp_logger in trainer.loggers:
        exp_logger.log_hyperparams(hparams)
