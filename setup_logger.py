"""Logger configuration."""
import logging

if __name__ == "__main__":

    def setup_logger() -> None:
        """Logger config setup."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler()],
        )
