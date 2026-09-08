"""Shared launcher for the validated full-study drivers."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "src" / "reference"
LDPC = ROOT / "src" / "ldpc"


def launch(module_name: str, default_arguments: list[str], required_arguments: list[str] | None = None) -> None:
    for path in (REFERENCE, LDPC, ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    arguments = list(sys.argv[1:]) or list(default_arguments)
    for option in required_arguments or []:
        flag, value = option.split("=", 1)
        if flag not in arguments:
            arguments.extend([flag, value])
    sys.argv = [sys.argv[0], *arguments]
    module = importlib.import_module(module_name)
    module.main()
