import logging

from lightning_utilities.core.rank_zero import rank_prefixed_message, rank_zero_only


class RankedLogger(logging.LoggerAdapter):
    """A multi-GPU-friendly python command line logger."""

    def __init__(
        self,
        name: str,
    ) -> None:
        super().__init__(logger=logging.getLogger(name))

    @rank_zero_only
    def log(self, level: int, msg: str) -> None:
        if not self.isEnabledFor(level):
            return

        rank = getattr(rank_zero_only, "rank", None)

        assert rank == 0, "Expect rank to be 0 when using rank_zero_only"

        msg = rank_prefixed_message(msg, rank)
        self.logger.log(level, msg)
