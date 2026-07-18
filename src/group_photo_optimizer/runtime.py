from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Optional


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def executable_directory() -> Path:
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path.cwd().resolve()


def prepare_frozen_working_directory() -> None:
    if is_frozen():
        os.chdir(str(executable_directory()))


def resolve_config_path(requested: Optional[Path]) -> Path:
    if requested is not None:
        return requested.resolve()
    root = executable_directory()
    destination = root / "config.yaml"
    if destination.exists():
        return destination
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        bundled = Path(bundle_root) / "config.yaml"
        if bundled.exists():
            try:
                shutil.copyfile(bundled, destination)
                return destination
            except OSError:
                # This normally only occurs when the EXE is placed in a read-only folder.
                raise RuntimeError(
                    "Cannot create config.yaml next to the EXE. Move the EXE to a writable folder."
                )
    return destination
