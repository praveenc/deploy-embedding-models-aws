import subprocess
import os
import shutil
from IPython.display import clear_output
from rich import print
import time
from datetime import datetime
from uuid import uuid4
from pathlib import Path


def print_container_logs(container_name: str, wait_seconds=60, log_lines=50):
    # Query the container ID using its name
    try:
        command = f"docker ps -q -f name={container_name}"
        container_id = subprocess.check_output(command, shell=True, text=True).strip()
    except subprocess.CalledProcessError:
        container_id = ""

    if not container_id:
        print(f"Container with name: {container_name} is not running!.")
    else:
        while True:
            # Check if the container is still running
            try:
                command = f"docker ps -q -f id={container_id}"
                output = subprocess.check_output(command, shell=True, text=True)
            except subprocess.CalledProcessError:
                output = ""

            if not output.strip():
                print("Container is not running. Stopping loop.")
                break

            # Fetch and display the last 50 lines of the Docker container logs
            log_command = f"docker logs {container_id} --tail {log_lines}"
            log_output = subprocess.check_output(log_command, shell=True, text=True)
            print(
                f"Last {log_lines} lines of container [i magenta3]{container_name}[/i magenta3] logs:"
            )
            print(f"Waiting {wait_seconds} secs ...")
            print(log_output)

            # Wait for a few seconds before fetching logs again
            clear_output(wait=True)
            time.sleep(wait_seconds)


def clear_ipynb_dirs(root_dir: Path):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Check if '.ipynb_checkpoints' exists in the list of directories
        for dirname in [".ipynb_checkpoints", "__pycache__"]:
            if dirname in dirnames:
                # Create the full path to the directory
                dir_to_delete = os.path.join(dirpath, dirname)
                # Print or log the path for verification
                print(f"Deleting: {dir_to_delete}")
                # Delete the directory
                shutil.rmtree(dir_to_delete)


def get_suffix():
    suffix = f"{str(uuid4())[:5]}-{datetime.now().strftime('%d%b%Y')}"
    return suffix


def units_to_bytes(value, unit="MB"):
    if unit == "MB":
        return int(value * 10**6)
    elif unit == "GB":
        return int(value * 10**9)
    elif unit == "MiB":
        return int(value * 1024**2)
    elif unit == "GiB":
        return int(value * 1024**3)
    else:
        raise ValueError("Unsupported unit. Please use 'MB', 'GB', 'MiB', or 'GiB'.")


def bytes_to_units(bytes_value, unit="MB"):
    if unit == "MB":
        return bytes_value / (10**6)
    elif unit == "GB":
        return bytes_value / (10**9)
    elif unit == "MiB":
        return bytes_value / (1024**2)
    elif unit == "GiB":
        return bytes_value / (1024**3)
    else:
        raise ValueError("Unsupported unit. Please use 'MB', 'GB', 'MiB', or 'GiB'.")
