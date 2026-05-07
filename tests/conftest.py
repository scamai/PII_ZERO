"""pytest configuration: subprocess isolation for crypto tests.

The `cryptography` Fernet cipher segfaults when run after presidio_analyzer
loads torch + its bundled OpenSSL into the same process (OpenSSL symbol
collision). Tests marked with @pytest.mark.subprocess are re-run in a fresh
Python interpreter via subprocess.run, bypassing the collision entirely.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "subprocess: run this test class/function in a fresh subprocess to avoid C-extension conflicts",
    )


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_call(item):
    if not item.get_closest_marker("subprocess"):
        return  # normal execution

    # Re-run just this test in a fresh interpreter.
    node_id = item.nodeid
    result = subprocess.run(
        [sys.executable, "-m", "pytest", node_id, "-x", "--no-header", "-q"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise AssertionError(
            f"Subprocess test failed:\n{result.stdout}\n{result.stderr}"
        )

    # Skip the in-process execution — test already passed in subprocess.
    pytest.skip(f"[subprocess] passed in child process")
