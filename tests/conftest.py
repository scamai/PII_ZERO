"""pytest configuration: GPU safety + subprocess isolation for crypto tests."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

# Disable CUDA for ALL tests — GPU driver state corruption caused full machine
# reboots in earlier sessions. Never let test collection touch the GPU.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

# Guard against infinite subprocess recursion: if this env var is set, we are
# already inside a child pytest process spawned by the hook below.
_IN_SUBPROCESS = os.environ.get("_PYTEST_SUBPROCESS_WORKER") == "1"


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "fast: pure-Python tests, no ML imports (<5s total)",
    )
    config.addinivalue_line(
        "markers",
        "slow: loads torch/spacy/paddle/presidio — run in CI or Docker only",
    )
    config.addinivalue_line(
        "markers",
        "subprocess: run in a fresh interpreter to avoid C-extension conflicts",
    )


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_call(item):
    # Skip hook if we are already inside a spawned subprocess (breaks recursion).
    if _IN_SUBPROCESS:
        return
    if not item.get_closest_marker("subprocess"):
        return

    node_id = item.nodeid
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "_PYTEST_SUBPROCESS_WORKER": "1"}
    result = subprocess.run(
        [sys.executable, "-m", "pytest", node_id, "-x", "--no-header", "-q"],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )

    if result.returncode != 0:
        raise AssertionError(
            f"Subprocess test failed:\n{result.stdout}\n{result.stderr}"
        )

    pytest.skip("[subprocess] passed in child process")
