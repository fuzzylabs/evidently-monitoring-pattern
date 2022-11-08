import os
import subprocess
from setup_logger import logging


def check_docker_installation():
    """
    Check if docker is installed.
    """
    logging.info("Check docker version")
    docker_version_result = os.system("docker -v")

    if docker_version_result:
        exit("Docker was not found. Try to install it with https://www.docker.com")


def run_docker_compose():
    """
    Run all containers using docker compose.
    """
    if os.system("docker image ls -q") != None:
        os.system("docker image rm $(docker image ls -q)")
    if os.system("docker volume ls -q") != None:
        os.system("docker volume rm $(docker volume ls -q)")
    logging.info("Running docker compose")
    run_script(cmd=["docker", "compose", "up", "-d"], wait=True)


def run_script(cmd: list, wait: bool) -> None:
    logging.info("Run %s", " ".join(cmd))
    script_process = subprocess.Popen(" ".join(cmd), stdout=subprocess.PIPE, shell=True)

    if wait:
        script_process.wait()

        if script_process.returncode != 0:
            exit(script_process.returncode)


def stop_docker_compose():
    os.system("docker-compose down")
    run_script(cmd=["docker", "volume", "rm $(docker volume ls -q)"], wait=True)
