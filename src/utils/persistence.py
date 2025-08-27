from pathlib import Path

import rich
import rich.syntax
import rich.tree
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf
from rich.console import Console

from utils.ranked_logger import RankedLogger

log = RankedLogger(__name__)


def _create_config_tree(cfg: DictConfig) -> rich.tree.Tree:
    tree = rich.tree.Tree("CONFIG")
    for field in cfg:
        field_str = str(field)
        branch = tree.add(field_str)
        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=True)
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

    log.info(f"Tags: {tags}")

    tags_path = Path(cfg.paths.tags_path)
    with tags_path.open("w") as file:
        rich.print(cfg.tags, file=file)
