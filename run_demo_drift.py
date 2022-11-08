"""Running a demo with drift."""
import logging
import os
import subprocess
import time

from setup_logger import setup_logger


def check_docker_installation() -> None:
    """Check if docker is installed."""
    logging.info("Check docker version")
    docker_version_result = os.system("docker -v")

    if docker_version_result:
        exit(
            "Docker was not found. Try to install it with https://www.docker.com"
        )


def check_dataset() -> None:
    """Check if dataset has been downloaded and prepared."""
    dataset_path = "datasets/house_price_random_forest"

    if not os.path.exists(dataset_path):
        logging.error(f"Dataset do not exist in path: {dataset_path}")


def run_docker_compose() -> None:
    """Run all containers using docker compose."""
    if os.system("docker image ls -q") is not None:
        os.system("docker image rm $(docker image ls -q)")
    if os.system("docker volume ls -q") is not None:
        os.system("docker volume rm $(docker volume ls -q)")

    logging.info("Running docker compose")

    run_script(cmd=["docker", "compose", "up", "-d"], wait=True)


def run_script(cmd: list, wait: bool) -> None:
    """Run command in a terminal.

    Args:
        cmd (list): commands to run
        wait (bool): wait for command to finish running or not
    """
    logging.info("Run %s", " ".join(cmd))
    script_process = subprocess.Popen(
        " ".join(cmd), stdout=subprocess.PIPE, shell=True
    )

    if wait:
        script_process.wait()

        if script_process.returncode != 0:
            exit(script_process.returncode)


def send_data_to_model_server() -> None:
    """Send data with drift to the model server for predictions."""
    os.system("python scenarios/drift.py")


def stop_docker_compose() -> None:
    """Stopping docker compose."""
    os.system("docker compose down")

    run_script(cmd=["docker", "volume", "rm $(docker volume ls -q)"], wait=True)


if __name__ == "__main__":
    setup_logger()
    check_docker_installation()
    check_dataset()
    run_docker_compose()
    time.sleep(5)
    send_data_to_model_server()
