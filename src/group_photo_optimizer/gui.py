from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import AppConfig
from .pipeline import PipelineResult, run_pipeline
from .runtime import executable_directory, resolve_config_path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


class GuiApi:
    def __init__(self, config_path: Path):
        self._config_path = config_path
        self._window = None
        self._lock = threading.Lock()
        self._lines: List[str] = []
        self._state = "idle"
        self._phase = "ready"
        self._error = ""
        self._result: Optional[PipelineResult] = None
        self._detection_done = 0
        self._image_count = 0

    def attach_window(self, window) -> None:
        self._window = window

    def initial_state(self) -> Dict[str, Any]:
        config = AppConfig.load(self._config_path)
        paths = config.resolve_paths(self._config_path)
        return {
            "input_dir": str(paths.input_dir),
            "output_dir": str(paths.output_dir),
            "config_path": str(self._config_path),
        }

    def select_folder(self) -> Optional[str]:
        import webview

        if self._window is None:
            return None
        selected = self._window.create_file_dialog(webview.FOLDER_DIALOG)
        if not selected:
            return None
        return str(selected[0])

    def start_processing(
        self, input_dir: str, reuse_analysis: bool, analyze_only: bool
    ) -> Dict[str, Any]:
        folder = Path(input_dir).expanduser()
        if not folder.is_dir():
            return {"ok": False, "error": "图片文件夹不存在"}
        with self._lock:
            if self._state == "running":
                return {"ok": False, "error": "任务正在运行"}
            self._lines = []
            self._state = "running"
            self._phase = "starting"
            self._error = ""
            self._result = None
            self._detection_done = 0
            self._image_count = sum(
                path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
                for path in folder.iterdir()
            )
        worker = threading.Thread(
            target=self._run,
            args=(folder, bool(reuse_analysis), bool(analyze_only)),
            daemon=True,
            name="group-photo-optimizer-worker",
        )
        worker.start()
        return {"ok": True}

    def _on_log(self, line: str) -> None:
        phase = self._phase
        if "DETECTION image=" in line:
            phase = "detecting"
        elif "ALIGNMENT image=" in line:
            phase = "aligning"
        elif "QUALITY_RANKING_BEGIN" in line:
            phase = "ranking"
        elif "REPLACEMENT_LIST_BEGIN" in line:
            phase = "replacing"
        elif "stage=output_writing" in line:
            phase = "writing"
        with self._lock:
            self._lines.append(line)
            self._phase = phase
            if "DETECTION image=" in line:
                self._detection_done += 1

    def _run(self, folder: Path, reuse_analysis: bool, analyze_only: bool) -> None:
        try:
            result = run_pipeline(
                self._config_path,
                reuse_analysis=reuse_analysis,
                analyze_only=analyze_only,
                input_dir_override=folder,
                log_callback=self._on_log,
            )
            with self._lock:
                self._result = result
                self._state = "completed"
                self._phase = "completed"
        except Exception as error:
            with self._lock:
                self._state = "failed"
                self._phase = "failed"
                self._error = str(error)

    def get_status(self, after: int = 0) -> Dict[str, Any]:
        with self._lock:
            start = max(0, min(int(after), len(self._lines)))
            result = self._result
            payload: Dict[str, Any] = {
                "state": self._state,
                "phase": self._phase,
                "error": self._error,
                "lines": self._lines[start:],
                "next": len(self._lines),
                "detection_done": self._detection_done,
                "image_count": self._image_count,
            }
            if result is not None:
                payload["result"] = {
                    "output_dir": str(result.output_dir),
                    "final_path": str(result.final_path),
                    "report_path": str(result.report_path),
                    "report_url": result.report_path.resolve().as_uri(),
                    "log_path": str(result.log_path),
                    "total_seconds": result.total_seconds,
                    "base_image": result.base_image,
                    "replaced_count": result.replaced_count,
                    "skipped_count": result.skipped_count,
                }
            return payload

    def open_output(self) -> bool:
        with self._lock:
            path = None if self._result is None else self._result.output_dir
        if path is None or not path.exists():
            return False
        os.startfile(str(path))
        return True

    def open_final(self) -> bool:
        with self._lock:
            path = None if self._result is None else self._result.final_path
        if path is None or not path.exists():
            return False
        os.startfile(str(path))
        return True

    def open_config(self) -> bool:
        if not self._config_path.exists():
            return False
        os.startfile(str(self._config_path))
        return True


def launch_gui() -> None:
    import sys
    import webview

    config_path = resolve_config_path(None)
    api = GuiApi(config_path)
    bundle_root = Path(getattr(sys, "_MEIPASS", executable_directory()))
    page = bundle_root / "gui" / "index.html"
    window = webview.create_window(
        "全家福优化器",
        url=page.resolve().as_uri(),
        js_api=api,
        width=1280,
        height=840,
        min_size=(960, 680),
        background_color="#f4f6f8",
    )
    api.attach_window(window)
    webview.start(gui="edgechromium", debug=False, private_mode=False)
