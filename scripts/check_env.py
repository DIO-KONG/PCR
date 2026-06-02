from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path


REQUIRED = ("numpy", "yaml")
OPTIONAL = ("open3d", "small_gicp")
PIP_PACKAGES = {"yaml": "PyYAML", "numpy": "numpy", "open3d": "open3d", "small_gicp": "small-gicp"}


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def install_packages(packages: list[str], offline: bool) -> None:
    if not packages:
        return
    command = [sys.executable, "-m", "pip", "install"]
    if offline:
        command.extend(["--no-index", "--find-links", str(Path("dependencies").resolve())])
    command.extend(packages)
    subprocess.check_call(command)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check or install PCR Python dependencies.")
    parser.add_argument("--install", action="store_true", help="Install missing required and optional dependencies.")
    parser.add_argument("--offline", action="store_true", help="Install only from dependencies/ wheels.")
    args = parser.parse_args()

    missing_required = [name for name in REQUIRED if not module_available(name)]
    missing_optional = [name for name in OPTIONAL if not module_available(name)]

    print(f"Python: {sys.executable}")
    print(f"Missing required: {missing_required or 'none'}")
    print(f"Missing optional: {missing_optional or 'none'}")

    if args.install:
        install_packages([PIP_PACKAGES[name] for name in missing_required + missing_optional], args.offline)
        return 0

    return 1 if missing_required else 0


if __name__ == "__main__":
    raise SystemExit(main())
