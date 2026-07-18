from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional


class CallbackHandler(logging.Handler):
    def __init__(self, callback: Callable[[str], None], formatter: logging.Formatter):
        super().__init__()
        self.callback = callback
        self.setFormatter(formatter)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.callback(self.format(record))
        except Exception:
            self.handleError(record)


@dataclass
class RunLog:
    logger: logging.Logger
    path: Path
    started_at: float = field(default_factory=time.perf_counter)
    stages: Dict[str, float] = field(default_factory=dict)

    def stage(self, name: str, started_at: float) -> float:
        elapsed = time.perf_counter() - started_at
        self.stages[name] = self.stages.get(name, 0.0) + elapsed
        self.logger.info("TIMING stage=%s seconds=%.3f", name, elapsed)
        return elapsed

    def finish(self, output_dir: Path) -> float:
        elapsed = time.perf_counter() - self.started_at
        self.logger.info("TIMING total_seconds=%.3f", elapsed)
        self.logger.info("RUN completed log=%s", self.path)
        for handler in self.logger.handlers:
            handler.flush()
        shutil.copyfile(self.path, output_dir / "latest.log")
        return elapsed

    def fail(self, output_dir: Path) -> None:
        for handler in self.logger.handlers:
            handler.flush()
        if self.path.exists():
            shutil.copyfile(self.path, output_dir / "latest.log")


def create_run_log(
    output_dir: Path, callback: Optional[Callable[[str], None]] = None
) -> RunLog:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / "run_{}.log".format(timestamp)
    logger = logging.getLogger("group_photo_optimizer")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(str(path), mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if callback is not None:
        logger.addHandler(CallbackHandler(callback, formatter))
    return RunLog(logger=logger, path=path)
