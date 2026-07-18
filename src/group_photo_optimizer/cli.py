from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import AppConfig
from .models import resolve_models
from .pipeline import run_pipeline
from .runtime import is_frozen, prepare_frozen_working_directory, resolve_config_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="group-photo-optimize",
        description="Select and conservatively repair the best photo from a fixed-position group burst.",
    )
    subparsers = parser.add_subparsers(dest="command")
    run = subparsers.add_parser("run", help="analyze, select, and blend")
    run.add_argument("--config", type=Path, default=None)
    run.add_argument("--reuse-analysis", action="store_true", help="reuse output/analysis_cache.pkl")
    run.add_argument("--analyze-only", action="store_true", help="select a base without replacing features")
    download = subparsers.add_parser("download-models", help="download the three local model files")
    download.add_argument("--config", type=Path, default=None)
    subparsers.add_parser("gui", help="open the desktop interface")
    return parser


def main() -> None:
    if is_frozen() and len(sys.argv) == 1:
        prepare_frozen_working_directory()
        from .gui import launch_gui

        launch_gui()
        return
    parser = build_parser()
    arguments = parser.parse_args()
    if getattr(arguments, "config", None) is None:
        prepare_frozen_working_directory()
    command = arguments.command or "run"
    if command == "gui":
        prepare_frozen_working_directory()
        from .gui import launch_gui

        launch_gui()
        return
    if command == "download-models":
        config_path = resolve_config_path(arguments.config)
        config = AppConfig.load(config_path)
        paths = config.resolve_paths(config_path)
        for name, path in resolve_models(paths.models_dir).items():
            print("{}: {}".format(name, path))
        return
    if command == "run":
        config_path = resolve_config_path(getattr(arguments, "config", None))
        run_pipeline(
            config_path,
            reuse_analysis=getattr(arguments, "reuse_analysis", False),
            analyze_only=getattr(arguments, "analyze_only", False),
        )
        return
    parser.error("unknown command")
