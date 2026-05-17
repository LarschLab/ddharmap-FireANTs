# Copyright (c) 2026 Rohit Jena. All rights reserved.
#
# This file is part of FireANTs, distributed under the terms of
# the FireANTs License version 1.0. A copy of the license can be found
# in the LICENSE file at the root of this repository.

import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import SimpleITK as sitk
import torch

from fireants.gui.runner import RegistrationJob, RegistrationSettings, run_batch_registration

try:
    from PySide6.QtCore import QObject, QSettings, Qt, QThread, Signal
    from PySide6.QtGui import QAction, QImage, QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QFileDialog,
        QFormLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QProgressBar,
        QSplitter,
        QStatusBar,
        QToolBar,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover - exercised only without optional deps.
    raise SystemExit("PySide6 is required for fireants-gui. Install FireANTs with the gui extra.") from exc


IMAGE_FILTER = "Images (*.nii *.nii.gz *.nrrd *.mha *.mhd *.tif *.tiff);;All files (*)"


@dataclass
class GuiJob:
    moving_bridge: Path
    warp_files: List[Path] = field(default_factory=list)
    status: str = "Queued"

    def display_text(self) -> str:
        return f"{self.status} | {self.moving_bridge.name} | warp files: {len(self.all_warp_files())}"

    def all_warp_files(self) -> List[Path]:
        paths = [self.moving_bridge] + list(self.warp_files)
        deduped = []
        seen = set()
        for path in paths:
            key = str(path)
            if key not in seen:
                seen.add(key)
                deduped.append(path)
        return deduped


class RegistrationWorker(QObject):
    progress = Signal(dict)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, fixed_bridge: Path, jobs: List[RegistrationJob], output_dir: Path, settings: RegistrationSettings):
        super().__init__()
        self.fixed_bridge = fixed_bridge
        self.jobs = jobs
        self.output_dir = output_dir
        self.settings = settings
        self._cancel_requested = False

    def cancel(self) -> None:
        self._cancel_requested = True

    def run(self) -> None:
        try:
            run_batch_registration(
                self.fixed_bridge,
                self.jobs,
                self.output_dir,
                settings=self.settings,
                progress_callback=self.progress.emit,
                cancel_requested=lambda: self._cancel_requested,
            )
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("FireANTs Bridge Registration")
        self.resize(1180, 760)
        self.settings_store = QSettings("FireANTs", "FireANTsGUI")
        self.jobs: List[GuiJob] = []
        self.worker_thread = None
        self.worker = None
        self._build_ui()
        self._restore_settings()
        self._set_running(False)

    def _build_ui(self) -> None:
        toolbar = QToolBar("Files")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        style = self.style()
        self.add_fixed_action = QAction(style.standardIcon(style.StandardPixmap.SP_DialogOpenButton), "Fixed Bridge", self)
        self.add_fixed_action.triggered.connect(self.choose_fixed_bridge)
        toolbar.addAction(self.add_fixed_action)

        self.add_moving_action = QAction(style.standardIcon(style.StandardPixmap.SP_FileDialogNewFolder), "Add Moving Bridges", self)
        self.add_moving_action.triggered.connect(self.add_moving_bridges)
        toolbar.addAction(self.add_moving_action)

        self.add_payload_action = QAction(style.standardIcon(style.StandardPixmap.SP_FileIcon), "Attach Warp Files", self)
        self.add_payload_action.triggered.connect(self.attach_warp_files)
        toolbar.addAction(self.add_payload_action)

        self.remove_action = QAction(style.standardIcon(style.StandardPixmap.SP_TrashIcon), "Remove Row", self)
        self.remove_action.triggered.connect(self.remove_selected_row)
        toolbar.addAction(self.remove_action)

        toolbar.addSeparator()
        self.start_action = QAction(style.standardIcon(style.StandardPixmap.SP_MediaPlay), "Start", self)
        self.start_action.triggered.connect(self.start_registration)
        toolbar.addAction(self.start_action)
        self.stop_action = QAction(style.standardIcon(style.StandardPixmap.SP_MediaStop), "Stop", self)
        self.stop_action.triggered.connect(self.stop_registration)
        toolbar.addAction(self.stop_action)

        root = QWidget()
        root_layout = QVBoxLayout(root)

        controls = QFormLayout()
        self.fixed_path_edit = QLineEdit()
        self.fixed_path_edit.setReadOnly(True)
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setReadOnly(True)
        self.device_edit = QLineEdit(default_device())
        choose_output = QPushButton("Choose Output")
        choose_output.clicked.connect(self.choose_output_dir)
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_path_edit, 1)
        output_row.addWidget(choose_output)
        controls.addRow("Fixed bridge", self.fixed_path_edit)
        controls.addRow("Output folder", output_row)
        controls.addRow("Device", self.device_edit)
        root_layout.addLayout(controls)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.job_list = QListWidget()
        self.job_list.currentRowChanged.connect(self.update_preview)
        splitter.addWidget(self.job_list)

        preview = QWidget()
        preview_layout = QHBoxLayout(preview)
        self.moving_preview = QLabel("No moving bridge selected")
        self.fixed_preview = QLabel("No fixed bridge selected")
        for label in (self.moving_preview, self.fixed_preview):
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setMinimumSize(360, 360)
            label.setStyleSheet("QLabel { background: #101418; color: #d9e2ec; border: 1px solid #323b44; }")
            label.setScaledContents(False)
        preview_layout.addWidget(self.moving_preview, 1)
        preview_layout.addWidget(self.fixed_preview, 1)
        splitter.addWidget(preview)
        splitter.setSizes([330, 850])
        root_layout.addWidget(splitter, 1)

        self.current_progress = QProgressBar()
        self.overall_progress = QProgressBar()
        self.current_progress.setRange(0, 100)
        self.overall_progress.setRange(0, 100)
        root_layout.addWidget(QLabel("Current stack"))
        root_layout.addWidget(self.current_progress)
        root_layout.addWidget(QLabel("Batch"))
        root_layout.addWidget(self.overall_progress)

        self.setCentralWidget(root)
        self.setStatusBar(QStatusBar())

    def _restore_settings(self) -> None:
        fixed = self.settings_store.value("fixed_bridge", "")
        output = self.settings_store.value("output_dir", "")
        if fixed:
            self.fixed_path_edit.setText(str(fixed))
        if output:
            self.output_path_edit.setText(str(output))
        self.update_preview()

    def choose_fixed_bridge(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose fixed bridge stack", "", IMAGE_FILTER)
        if not path:
            return
        self.fixed_path_edit.setText(path)
        self.settings_store.setValue("fixed_bridge", path)
        self.update_preview()

    def choose_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose output folder")
        if not path:
            return
        self.output_path_edit.setText(path)
        self.settings_store.setValue("output_dir", path)

    def add_moving_bridges(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Choose moving bridge stacks", "", IMAGE_FILTER)
        for raw_path in paths:
            path = Path(raw_path)
            self.jobs.append(GuiJob(moving_bridge=path))
        self.refresh_job_list()
        if paths and self.job_list.currentRow() < 0:
            self.job_list.setCurrentRow(0)

    def attach_warp_files(self) -> None:
        row = self.job_list.currentRow()
        if row < 0:
            QMessageBox.information(self, "No row selected", "Select one moving bridge row first.")
            return
        paths, _ = QFileDialog.getOpenFileNames(self, "Choose moving-side files to warp", "", IMAGE_FILTER)
        for raw_path in paths:
            path = Path(raw_path)
            if path not in self.jobs[row].warp_files and path != self.jobs[row].moving_bridge:
                self.jobs[row].warp_files.append(path)
        self.refresh_job_list(keep_row=row)

    def remove_selected_row(self) -> None:
        row = self.job_list.currentRow()
        if row < 0:
            return
        del self.jobs[row]
        self.refresh_job_list(keep_row=min(row, len(self.jobs) - 1))

    def refresh_job_list(self, keep_row: int = -1) -> None:
        self.job_list.clear()
        for job in self.jobs:
            item = QListWidgetItem(job.display_text())
            item.setToolTip("\n".join(str(path) for path in job.all_warp_files()))
            self.job_list.addItem(item)
        if keep_row >= 0 and self.jobs:
            self.job_list.setCurrentRow(keep_row)

    def update_preview(self) -> None:
        fixed_path = self.fixed_path_edit.text()
        if fixed_path:
            self._set_preview(self.fixed_preview, Path(fixed_path), "Fixed bridge")
        else:
            self.fixed_preview.setText("No fixed bridge selected")

        row = self.job_list.currentRow()
        if row >= 0 and row < len(self.jobs):
            self._set_preview(self.moving_preview, self.jobs[row].moving_bridge, "Moving bridge")
        else:
            self.moving_preview.setText("No moving bridge selected")

    def _set_preview(self, label: QLabel, path: Path, title: str) -> None:
        try:
            pixmap = image_preview_pixmap(path, max_side=420)
        except Exception as exc:
            label.setPixmap(QPixmap())
            label.setText(f"{title}\n{path.name}\n{exc}")
            return
        label.setPixmap(pixmap)
        label.setToolTip(str(path))

    def start_registration(self) -> None:
        if not self.fixed_path_edit.text() or not self.output_path_edit.text() or not self.jobs:
            QMessageBox.warning(self, "Missing inputs", "Choose a fixed bridge, output folder, and at least one moving bridge.")
            return

        self.settings_store.setValue("fixed_bridge", self.fixed_path_edit.text())
        self.settings_store.setValue("output_dir", self.output_path_edit.text())
        settings = RegistrationSettings.default()
        settings.device = self.device_edit.text().strip() or settings.device
        settings.progress_bar = False
        run_output_dir = make_run_output_dir(Path(self.output_path_edit.text()))
        settings.metrics_dir = run_output_dir / "_metrics"
        runner_jobs = [RegistrationJob(job.moving_bridge, job.warp_files) for job in self.jobs]

        self.worker_thread = QThread(self)
        self.worker = RegistrationWorker(Path(self.fixed_path_edit.text()), runner_jobs, run_output_dir, settings)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.handle_progress)
        self.worker.failed.connect(self.handle_failure)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(lambda: self._set_running(False))
        self.worker_thread.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(self._clear_worker_refs)
        self._set_running(True)
        self.statusBar().showMessage(f"Running in {run_output_dir}")
        self.worker_thread.start()

    def stop_registration(self) -> None:
        if self.worker is not None:
            self.worker.cancel()
            self.statusBar().showMessage("Stopping after the current operation...")

    def handle_progress(self, event: Dict[str, object]) -> None:
        name = str(event.get("event", ""))
        row_index = int(event.get("row_index", -1)) if "row_index" in event else -1
        if row_index >= 0 and row_index < len(self.jobs):
            if name == "row_start":
                self.jobs[row_index].status = "Running"
            elif name == "row_complete":
                self.jobs[row_index].status = "Done"
            self.refresh_job_list(keep_row=row_index)

        completed = int(event.get("completed_units", 0))
        total = int(event.get("total_units", 0))
        row_completed = int(event.get("row_completed_units", 0))
        row_total = int(event.get("row_total_units", 0))
        if total > 0:
            self.overall_progress.setValue(int(completed * 100 / total))
        if row_total > 0:
            self.current_progress.setValue(int(row_completed * 100 / row_total))

        message = self._status_message(event)
        if message:
            self.statusBar().showMessage(message)

    def handle_failure(self, message: str) -> None:
        self.statusBar().showMessage(message)
        QMessageBox.critical(self, "Registration failed", message)

    def _status_message(self, event: Dict[str, object]) -> str:
        name = str(event.get("event", ""))
        row_index = event.get("row_index")
        prefix = f"Stack {int(row_index) + 1}/{len(self.jobs)}: " if row_index is not None else ""
        if name == "registration_progress":
            stage = event.get("stage", "Registration")
            registration_event = event.get("registration_event")
            if registration_event != "iteration":
                return f"{prefix}{stage} {str(registration_event).replace('_', ' ')}"
            scale = event.get("scale", "")
            iteration = event.get("iteration", "")
            iterations = event.get("iterations", "")
            loss = event.get("loss")
            loss_text = f", loss {float(loss):.5f}" if isinstance(loss, (float, int)) else ""
            return f"{prefix}{stage} scale {scale} iter {iteration}/{iterations}{loss_text}"
        if name == "warp_start":
            return f"{prefix}Warping {Path(str(event.get('path'))).name}"
        if name == "warp_complete":
            return f"{prefix}Saved {Path(str(event.get('output_path'))).name}"
        if name == "batch_start" and event.get("output_dir"):
            return f"Running in {event.get('output_dir')}"
        if name == "batch_complete":
            return "Batch complete"
        return ""

    def _set_running(self, running: bool) -> None:
        self.add_fixed_action.setEnabled(not running)
        self.add_moving_action.setEnabled(not running)
        self.add_payload_action.setEnabled(not running)
        self.remove_action.setEnabled(not running)
        self.start_action.setEnabled(not running)
        self.stop_action.setEnabled(running)
        if running:
            self.current_progress.setValue(0)
            self.overall_progress.setValue(0)
        else:
            if self.worker is None:
                self.statusBar().showMessage("Ready")

    def _clear_worker_refs(self) -> None:
        self.worker = None
        self.worker_thread = None


def image_preview_pixmap(path: Path, max_side: int = 420) -> QPixmap:
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)
    if array.ndim == 4:
        array = array[..., 0]
    if array.ndim == 3:
        array = array[array.shape[0] // 2]
    if array.ndim != 2:
        raise ValueError(f"Unsupported preview dimensions: {array.shape}")
    array = np.asarray(array, dtype=np.float32)
    lo, hi = np.percentile(array, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    array = np.clip((array - lo) / (hi - lo), 0.0, 1.0)
    array = (array * 255).astype(np.uint8)
    h, w = array.shape
    qimage = QImage(array.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
    pixmap = QPixmap.fromImage(qimage)
    return pixmap.scaled(max_side, max_side, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_run_output_dir(output_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(output_root) / f"fireants_gui_run_{timestamp}"


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
