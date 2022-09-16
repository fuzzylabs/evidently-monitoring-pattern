import os
import logging
import subprocess
import time


def setup_logger() -> None:
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )


def check_docker_installation():
    logging.info("Check docker version")
    docker_version_result = os.system("docker -v")

    if docker_version_result:
        exit("Docker was not found. Try to install it with https://www.docker.com")


def check_dataset():
    dataset_path = "datasets/house_price_random_forest"

    if not os.path.exists(dataset_path):
        logging.error(f"Dataset do not exist in path: {dataset_path}")


def run_docker_compose():
    logging.info("Running docker compose")
    run_script(cmd=["docker", "compose", "up", "-d"], wait=True)


def run_script(cmd: list, wait: bool) -> None:
    logging.info("Run %s", " ".join(cmd))
    script_process = subprocess.Popen(" ".join(cmd), stdout = subprocess.PIPE, shell = True)

    if wait:
        script_process.wait()

        if script_process.returncode != 0:
            exit(script_process.returncode)


def send_data_to_model_server():
    os.system("python scenarios/no_drift.py")


def stop_docker_compose():
    os.system("docker compose down")


if __name__ == "__main__":
    setup_logger()
    check_docker_installation()
    check_dataset()
    run_docker_compose()
    time.sleep(5)
    send_data_to_model_server()