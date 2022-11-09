"""Docker utility functions."""
import logging
import os
import subprocess


def check_docker_installation():
    """Check if docker is installed."""
    logging.info("Check docker version")
    docker_version_result = os.system("docker -v")

    if docker_version_result:
        exit(
            "Docker was not found. Try to install it with https://www.docker.com"
        )


def running_grafana_prometheus_docker_compose_services() -> bool:
    """Check if granfana and prometheus services are running in docker compose.

    Returns:
        bool:  if grafana and prometheus services are running in docker compose.
    """
    grafana_command = "docker compose ps -q grafana --status=running"
    prometheus_command = "docker compose ps -q prometheus --status=running"
    if (os.system(grafana_command) is not None) and (os.system(prometheus_command) is not None):  # fmt: skip
        # TODO: why 256 here?
        if (os.system(grafana_command) == 256 and os.system(prometheus_command) == 256):  # fmt: skip
            return False
        return True
    return False


def run_docker_compose():
    """Run all containers using docker compose."""
    # run docker compose only when grafana and prometheus services are not running
    if not running_grafana_prometheus_docker_compose_services():
        logging.info("Running docker compose")
        run_script(cmd=["docker", "compose", "up", "-d"], wait=True)
    else:
        logging.info(
            "Found services : ['prometheus', 'grafana'] already running..."
        )


def run_script(cmd: list, wait: bool) -> None:
    """Run script using subprocess.

    Args:
        cmd (list): List of commands to run
        wait (bool): Wait for the script to finish

    """
    logging.info("Run %s", " ".join(cmd))
    script_process = subprocess.Popen(
        " ".join(cmd), stdout=subprocess.PIPE, shell=True
    )

    if wait:
        script_process.wait()

        if script_process.returncode != 0:
            exit(script_process.returncode)


def stop_docker_compose():
    """Stop docker compose."""
    os.system("docker compose down")
    run_script(cmd=["docker", "volume", "rm $(docker volume ls -q)"], wait=True)
