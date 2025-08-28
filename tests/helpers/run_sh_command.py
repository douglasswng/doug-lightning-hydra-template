# Replace the current run_sh_command.py with:
import subprocess  # nosec: B404
import sys

import pytest


def run_sh_command(command: list[str]) -> None:
    """Execute shell commands using subprocess instead of sh library.

    :param command: A list of shell commands as strings.
    """
    try:
        subprocess.run(  # nosec: B603
            [sys.executable, *command], capture_output=True, text=True, check=True
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(reason=e.stderr)
