#!/usr/bin/env python3
"""Run the release-package validation suite without third-party test tooling."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"]
raise SystemExit(subprocess.call(command, cwd=ROOT))
