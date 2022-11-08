"""Logger configuration."""
import logging


def setup_logger() -> None:
    """Logger config setup."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )
