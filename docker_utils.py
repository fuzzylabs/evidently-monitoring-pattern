"""Docker utility functions."""
import logging
import os
import subprocess


def check_docker_installation():
    """Check if docker is installed."""
    logging.info("Check docker version")
    docker_version_result = os.system("docker -v")

    if docker_version_result:
        exit("Docker was not found. Try to install it with https://www.docker.com")


def run_docker_compose():
    """Run all containers using docker compose."""
    if os.system("docker image ls -q") is not None:
        os.system("docker image rm $(docker image ls -q)")
    if os.system("docker volume ls -q") is not None:
        os.system("docker volume rm $(docker volume ls -q)")
    logging.info("Running docker compose")
    run_script(cmd=["docker", "compose", "up", "-d"], wait=True)


def run_script(cmd: list, wait: bool) -> None:
    """Run script using subprocess.

    Args:
        cmd (list): List of commands to run
        wait (bool): Wait for the script to finish

    """
    logging.info("Run %s", " ".join(cmd))
    script_process = subprocess.Popen(" ".join(cmd), stdout=subprocess.PIPE, shell=True)

    if wait:
        script_process.wait()

        if script_process.returncode != 0:
            exit(script_process.returncode)


def stop_docker_compose():
    """Stop docker compose"""
    logging.info("Stopping docker compose")
    os.system("docker compose down")
    run_script(cmd=["docker", "volume", "rm $(docker volume ls -q)"], wait=True)
