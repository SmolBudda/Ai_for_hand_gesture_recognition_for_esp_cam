from __future__ import annotations

from rich.console import Console
from rich.logging import RichHandler
import logging


console = Console()


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )
    if not verbose:
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
