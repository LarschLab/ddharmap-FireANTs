# Copyright (c) 2026 Rohit Jena. All rights reserved.
#
# This file is part of FireANTs, distributed under the terms of
# the FireANTs License version 1.0. A copy of the license can be found
# in the LICENSE file at the root of this repository.

import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import SimpleITK as sitk
import torch

from fireants.gui.runner import (
    PIPELINE_ANTS_RIGID_AFFINE_SYN,
    REGISTRATION_PROFILE_NAMES,
    SUPPORTED_GREEDY_OPTIMIZERS,
    SUPPORTED_LOSS_TYPES,
    SUPPORTED_MOMENTS_ORIENTATIONS,
    SUPPORTED_SYN_OPTIMIZERS,
    RegistrationJob,
    RegistrationSettings,
    parse_float_list,
    parse_int_list,
    registration_settings_for_profile,
    run_batch_registration,
    validate_registration_settings,
)

try:
    from PySide6.QtCore import QObject, QSettings, Qt, QThread, Signal
    from PySide6.QtGui import QAction, QImage, QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QFileDialog,
        QFormLayout,
        QCheckBox,
        QComboBox,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QProgressBar,
        QScrollArea,
        QSplitter,
        QStatusBar,
        QTabWidget,
        QToolBar,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover - exercised only without optional deps.
    raise SystemExit("PySide6 is required for fireants-gui. Install FireANTs with the gui extra.") from exc


IMAGE_FILTER = "Images (*.nii *.nii.gz *.nrrd *.mha *.mhd *.tif *.tiff);;All files (*)"


def _set_help(widget, text: str) -> None:
    widget.setToolTip(text)
    if hasattr(widget, "setStatusTip"):
        widget.setStatusTip(text)


@dataclass
class GuiJob:
    moving_bridge: Path
    warp_files: List[Path] = field(default_factory=list)
    status: str = "Queued"
    live_preview: Optional[QPixmap] = None

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
        self._loading_profile_fields = False
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
        _set_help(self.add_fixed_action, "Choose the fixed reference bridge stack. Other bridge stacks will be aligned into this image space.")
        toolbar.addAction(self.add_fixed_action)

        self.add_moving_action = QAction(style.standardIcon(style.StandardPixmap.SP_FileDialogNewFolder), "Add Moving Bridges", self)
        self.add_moving_action.triggered.connect(self.add_moving_bridges)
        _set_help(self.add_moving_action, "Add one or more moving bridge stacks. Each moving stack is registered to the fixed bridge.")
        toolbar.addAction(self.add_moving_action)

        self.add_payload_action = QAction(style.standardIcon(style.StandardPixmap.SP_FileIcon), "Attach Warp Files", self)
        self.add_payload_action.triggered.connect(self.attach_warp_files)
        _set_help(self.add_payload_action, "Attach moving-side files that should receive the same final transform as the selected moving bridge.")
        toolbar.addAction(self.add_payload_action)

        self.remove_action = QAction(style.standardIcon(style.StandardPixmap.SP_TrashIcon), "Remove Row", self)
        self.remove_action.triggered.connect(self.remove_selected_row)
        _set_help(self.remove_action, "Remove the selected moving bridge row from the queue.")
        toolbar.addAction(self.remove_action)

        toolbar.addSeparator()
        self.start_action = QAction(style.standardIcon(style.StandardPixmap.SP_MediaPlay), "Start", self)
        self.start_action.triggered.connect(self.start_registration)
        _set_help(self.start_action, "Start registration for every queued moving bridge and write outputs into a new run folder.")
        toolbar.addAction(self.start_action)
        self.stop_action = QAction(style.standardIcon(style.StandardPixmap.SP_MediaStop), "Stop", self)
        self.stop_action.triggered.connect(self.stop_registration)
        _set_help(self.stop_action, "Request cancellation. FireANTs stops after the current operation finishes.")
        toolbar.addAction(self.stop_action)

        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(8, 6, 8, 6)
        root_layout.setSpacing(6)

        self.fixed_path_edit = QLineEdit()
        self.fixed_path_edit.setReadOnly(True)
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setReadOnly(True)
        self.device_edit = QLineEdit(default_device())
        self.device_edit.setMaximumWidth(150)
        choose_fixed = QPushButton("Browse")
        choose_fixed.clicked.connect(self.choose_fixed_bridge)
        choose_output = QPushButton("Browse")
        choose_output.clicked.connect(self.choose_output_dir)
        self.advanced_profile_button = QPushButton("Advanced")
        self.advanced_profile_button.clicked.connect(self.show_profile_tab)
        self.profile_combo = QComboBox()
        self.profile_combo.addItems(REGISTRATION_PROFILE_NAMES)
        self.profile_combo.currentTextChanged.connect(self.apply_profile_preset)
        _set_help(self.fixed_path_edit, "Fixed bridge path. This image defines the output space for registration.")
        _set_help(self.output_path_edit, "Output folder. Each run creates a timestamped child folder here.")
        _set_help(self.device_edit, "Compute device, such as cuda:0, mps, or cpu. GPU devices are faster but have stricter memory limits.")
        _set_help(choose_fixed, "Browse for the fixed reference bridge stack.")
        _set_help(choose_output, "Browse for the folder where registered outputs should be written.")
        _set_help(self.profile_combo, "Registration preset. Current Full is the full default; Memory Saver uses cheaper settings; Debug is a quick wiring check.")
        _set_help(self.advanced_profile_button, "Open the Profile tab to tune registration parameters.")

        fixed_row = QHBoxLayout()
        fixed_label = QLabel("Fixed")
        _set_help(fixed_label, self.fixed_path_edit.toolTip())
        fixed_row.addWidget(fixed_label)
        fixed_row.addWidget(self.fixed_path_edit, 1)
        fixed_row.addWidget(choose_fixed)
        output_row = QHBoxLayout()
        output_label = QLabel("Output")
        _set_help(output_label, self.output_path_edit.toolTip())
        output_row.addWidget(output_label)
        output_row.addWidget(self.output_path_edit, 1)
        output_row.addWidget(choose_output)
        device_label = QLabel("Device")
        _set_help(device_label, self.device_edit.toolTip())
        output_row.addWidget(device_label)
        output_row.addWidget(self.device_edit)
        profile_row = QHBoxLayout()
        profile_label = QLabel("Profile")
        _set_help(profile_label, self.profile_combo.toolTip())
        profile_row.addWidget(profile_label)
        profile_row.addWidget(self.profile_combo, 1)
        profile_row.addWidget(self.advanced_profile_button)
        root_layout.addLayout(fixed_row)
        root_layout.addLayout(output_row)
        root_layout.addLayout(profile_row)

        self.loss_combo = QComboBox()
        self.loss_combo.addItems(SUPPORTED_LOSS_TYPES)
        self.cc_kernel_edit = QLineEdit()
        self.winsorize_enabled_check = QCheckBox("Enabled")
        self.winsorize_lower_edit = QLineEdit()
        self.winsorize_upper_edit = QLineEdit()
        self.moments_scale_edit = QLineEdit()
        self.moments_order_edit = QLineEdit()
        self.moments_orientation_combo = QComboBox()
        self.moments_orientation_combo.addItems(SUPPORTED_MOMENTS_ORIENTATIONS)
        self.moments_scaling_check = QCheckBox("Enabled")
        self.rigid_loss_combo = QComboBox()
        self.rigid_loss_combo.addItems(SUPPORTED_LOSS_TYPES)
        self.rigid_mi_bins_edit = QLineEdit()
        self.rigid_scales_edit = QLineEdit()
        self.rigid_iterations_edit = QLineEdit()
        self.rigid_lr_edit = QLineEdit()
        self.affine_loss_combo = QComboBox()
        self.affine_loss_combo.addItems(SUPPORTED_LOSS_TYPES)
        self.affine_mi_bins_edit = QLineEdit()
        self.affine_scales_edit = QLineEdit()
        self.affine_iterations_edit = QLineEdit()
        self.affine_lr_edit = QLineEdit()
        self.greedy_scales_edit = QLineEdit()
        self.greedy_iterations_edit = QLineEdit()
        self.greedy_lr_edit = QLineEdit()
        self.greedy_optimizer_combo = QComboBox()
        self.greedy_optimizer_combo.addItems(SUPPORTED_GREEDY_OPTIMIZERS)
        self.greedy_reset_check = QCheckBox("Reset")
        self.greedy_offload_check = QCheckBox("Offload")
        _set_help(self.greedy_reset_check, "Clear Adam optimizer memory whenever Greedy changes resolution scale. This is more conservative but may lose useful momentum.")
        _set_help(self.greedy_offload_check, "Store Greedy Adam state on CPU. This can reduce GPU memory use at the cost of slower updates.")
        self.greedy_flags_widget = QWidget()
        _set_help(self.greedy_flags_widget, "Adam memory controls. Reset clears optimizer history between scales; Offload stores Adam state on CPU to reduce GPU memory.")
        greedy_flags = QHBoxLayout(self.greedy_flags_widget)
        greedy_flags.setContentsMargins(0, 0, 0, 0)
        greedy_flags.addWidget(self.greedy_reset_check)
        greedy_flags.addWidget(self.greedy_offload_check)
        greedy_flags.addStretch(1)
        self.syn_loss_combo = QComboBox()
        self.syn_loss_combo.addItems(SUPPORTED_LOSS_TYPES)
        self.syn_cc_kernel_edit = QLineEdit()
        self.syn_scales_edit = QLineEdit()
        self.syn_iterations_edit = QLineEdit()
        self.syn_lr_edit = QLineEdit()
        self.syn_optimizer_combo = QComboBox()
        self.syn_optimizer_combo.addItems(SUPPORTED_SYN_OPTIMIZERS)
        self.smooth_warp_sigma_edit = QLineEdit()
        self.smooth_grad_sigma_edit = QLineEdit()
        self.preview_max_side_edit = QLineEdit()
        self.preview_enabled_check = QCheckBox("Enabled")
        self._set_compact_profile_field_widths()
        self.profile_widgets = [
            self.profile_combo,
            self.advanced_profile_button,
            self.loss_combo,
            self.cc_kernel_edit,
            self.winsorize_enabled_check,
            self.winsorize_lower_edit,
            self.winsorize_upper_edit,
            self.moments_scale_edit,
            self.moments_order_edit,
            self.moments_orientation_combo,
            self.moments_scaling_check,
            self.rigid_loss_combo,
            self.rigid_mi_bins_edit,
            self.rigid_scales_edit,
            self.rigid_iterations_edit,
            self.rigid_lr_edit,
            self.affine_loss_combo,
            self.affine_mi_bins_edit,
            self.affine_scales_edit,
            self.affine_iterations_edit,
            self.affine_lr_edit,
            self.greedy_scales_edit,
            self.greedy_iterations_edit,
            self.greedy_lr_edit,
            self.greedy_optimizer_combo,
            self.greedy_reset_check,
            self.greedy_offload_check,
            self.syn_loss_combo,
            self.syn_cc_kernel_edit,
            self.syn_scales_edit,
            self.syn_iterations_edit,
            self.syn_lr_edit,
            self.syn_optimizer_combo,
            self.smooth_warp_sigma_edit,
            self.smooth_grad_sigma_edit,
            self.preview_enabled_check,
            self.preview_max_side_edit,
        ]

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_sidebar())
        splitter.addWidget(self._build_preview_area())
        splitter.setSizes([330, 850])
        root_layout.addWidget(splitter, 1)

        self.current_progress = QProgressBar()
        self.overall_progress = QProgressBar()
        self.current_progress.setRange(0, 100)
        self.overall_progress.setRange(0, 100)
        _set_help(self.current_progress, "Progress for the currently running moving bridge stack.")
        _set_help(self.overall_progress, "Progress across the full queued batch.")
        current_progress_label = QLabel("Current stack")
        batch_progress_label = QLabel("Batch")
        _set_help(current_progress_label, self.current_progress.toolTip())
        _set_help(batch_progress_label, self.overall_progress.toolTip())
        root_layout.addWidget(current_progress_label)
        root_layout.addWidget(self.current_progress)
        root_layout.addWidget(batch_progress_label)
        root_layout.addWidget(self.overall_progress)

        self.setCentralWidget(root)
        self.setStatusBar(QStatusBar())

    def _build_sidebar(self) -> QWidget:
        self.sidebar_tabs = QTabWidget()
        self.sidebar_tabs.setMinimumWidth(300)
        self.sidebar_tabs.setMaximumWidth(380)

        queue_tab = QWidget()
        queue_layout = QVBoxLayout(queue_tab)
        queue_layout.setContentsMargins(8, 8, 8, 8)
        self.job_list = QListWidget()
        self.job_list.currentRowChanged.connect(self.update_preview)
        _set_help(self.job_list, "Registration queue. Select a moving bridge row to preview it or attach additional moving-side files.")
        queue_layout.addWidget(self.job_list, 1)
        self.attach_payload_button = QPushButton("Attach warp files")
        self.attach_payload_button.clicked.connect(self.attach_warp_files)
        _set_help(self.attach_payload_button, "Attach extra moving-side files to the selected row. They are warped with the same final transform as the moving bridge.")
        queue_layout.addWidget(self.attach_payload_button)
        self.sidebar_tabs.addTab(queue_tab, "Queue")

        profile_scroll = QScrollArea()
        profile_scroll.setWidgetResizable(True)
        profile_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        profile_body = QWidget()
        profile_layout = QVBoxLayout(profile_body)
        profile_layout.setContentsMargins(8, 8, 8, 8)
        profile_layout.setSpacing(8)
        self.loss_group = self._profile_group(
            "Loss",
            [
                ("Type", self.loss_combo, "Similarity metric used to judge alignment. CC is a strong default for same-modality images; MI helps when intensities differ; fused versions use optimized kernels when available."),
                ("CC kernel", self.cc_kernel_edit, "Local window size for cross-correlation. Larger values are smoother and more stable; smaller values follow local detail but can be noisier. Use odd numbers."),
            ],
        )
        self.preprocess_group = self._profile_group(
            "Preprocess",
            [
                ("Winsorize", self.winsorize_enabled_check, "Clip extreme intensities before ANTs-like registration. This can reduce the effect of bright outliers."),
                ("Lower", self.winsorize_lower_edit, "Lower quantile for clipping. Higher values remove more dark outliers."),
                ("Upper", self.winsorize_upper_edit, "Upper quantile for clipping. Lower values remove more bright outliers."),
            ],
        )
        self.moments_group = self._profile_group(
            "Moments",
            [
                ("Scale", self.moments_scale_edit, "Downsample factor for the initial moments alignment. Larger values are faster and use less memory; smaller values use more image detail."),
                ("Order", self.moments_order_edit, "Moment order for initialization. 1 aligns centers only; 2 also aligns principal axes and is required for scaling correction."),
                ("Orientation", self.moments_orientation_combo, "Which principal-axis orientations to test. rot tests rotations, antirot tests mirrored candidates, and both keeps the best by the selected loss."),
                ("Scaling", self.moments_scaling_check, "Allow moments initialization to estimate scale differences. Helps when images are stretched or have spacing/size mismatch; requires order 2."),
            ],
        )
        self.rigid_group = self._profile_group(
            "Rigid",
            [
                ("Metric", self.rigid_loss_combo, "Similarity metric for rigid registration. MI is often safer for broad global alignment or different contrasts."),
                ("MI bins", self.rigid_mi_bins_edit, "Number of intensity bins for mutual information. More bins capture more intensity detail but can be noisier and slower."),
                ("Scales", self.rigid_scales_edit, "Coarse-to-fine downsample factors. Larger first values start with broad motion; ending near 1 uses full detail."),
                ("Iterations", self.rigid_iterations_edit, "Optimization steps at each rigid scale. More iterations can improve alignment but take longer."),
                ("LR", self.rigid_lr_edit, "Rigid learning rate. Larger values move faster but can overshoot; smaller values are slower and more cautious."),
            ],
        )
        self.affine_group = self._profile_group(
            "Affine",
            [
                ("Metric", self.affine_loss_combo, "Similarity metric for affine refinement. CC is good for matching intensity structure; MI helps across contrasts."),
                ("MI bins", self.affine_mi_bins_edit, "Number of intensity bins for affine mutual information. More bins add detail but cost memory/time."),
                ("Scales", self.affine_scales_edit, "Coarse-to-fine affine levels. Coarse scales capture big global differences before full-detail refinement."),
                ("Iterations", self.affine_iterations_edit, "Optimization steps at each affine scale. More steps can improve fit but increase runtime."),
                ("LR", self.affine_lr_edit, "Affine learning rate. Larger values update faster but can become unstable."),
            ],
        )
        self.greedy_group = self._profile_group(
            "Greedy",
            [
                ("Scales", self.greedy_scales_edit, "Coarse-to-fine deformable levels. Higher scales are cheaper and capture broad shape changes; lower scales add local detail."),
                ("Iterations", self.greedy_iterations_edit, "Deformable update steps at each Greedy scale. More steps can fit more detail but take longer."),
                ("LR", self.greedy_lr_edit, "Greedy warp learning rate. Larger values make bigger deformation updates; reduce it if the warp looks unstable."),
                ("Optimizer", self.greedy_optimizer_combo, "Warp optimizer. Adam adapts update size per location; SGD is simpler and can use less optimizer memory."),
                ("Adam state", self.greedy_flags_widget, "Adam memory controls. Reset clears optimizer history between scales; Offload stores Adam state on CPU to reduce GPU memory."),
            ],
        )
        self.syn_group = self._profile_group(
            "SyN",
            [
                ("Metric", self.syn_loss_combo, "Similarity metric for symmetric deformable registration. CC is the usual same-modality choice."),
                ("CC kernel", self.syn_cc_kernel_edit, "Local CC window for SyN. Larger windows are smoother; smaller windows follow finer detail."),
                ("Scales", self.syn_scales_edit, "Coarse-to-fine SyN levels. SyN moves both images toward a midpoint, so it is heavier than Greedy."),
                ("Iterations", self.syn_iterations_edit, "SyN update steps at each scale. More iterations can improve symmetric alignment but cost more time."),
                ("LR", self.syn_lr_edit, "SyN learning rate. SyN usually uses a smaller value than Greedy because it updates two warps."),
                ("Optimizer", self.syn_optimizer_combo, "SyN warp optimizer. Adam adapts per location; SGD is simpler and closer to the ANTs-like preset."),
            ],
        )
        self.preview_group = self._profile_group(
            "Preview",
            [
                ("Live", self.preview_enabled_check, "Show live magenta/green overlays after registration stages. Turning it off saves preview work."),
                ("Max side", self.preview_max_side_edit, "Largest preview image side in pixels. Larger previews show more detail but cost more memory/time."),
                ("Warp sigma", self.smooth_warp_sigma_edit, "Smooth the deformation field. Higher values make smoother, less flexible warps."),
                ("Grad sigma", self.smooth_grad_sigma_edit, "Smooth deformation gradients before updates. Higher values reduce noisy local motion."),
            ],
        )
        for group in (
            self.loss_group,
            self.preprocess_group,
            self.moments_group,
            self.rigid_group,
            self.affine_group,
            self.greedy_group,
            self.syn_group,
            self.preview_group,
        ):
            profile_layout.addWidget(group)
        profile_layout.addStretch(1)
        profile_scroll.setWidget(profile_body)
        self.sidebar_tabs.addTab(profile_scroll, "Profile")
        return self.sidebar_tabs

    def _profile_group(self, title: str, rows: List[tuple]) -> QGroupBox:
        group = QGroupBox(title)
        layout = QFormLayout(group)
        layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        for row in rows:
            label, widget = row[:2]
            label_widget = QLabel(label)
            if len(row) > 2:
                tooltip = row[2]
                _set_help(label_widget, tooltip)
                _set_help(widget, tooltip)
            layout.addRow(label_widget, widget)
        return group

    def _set_form_row_visible(self, widget: QWidget, visible: bool) -> None:
        widget.setVisible(visible)
        layout = widget.parentWidget().layout() if widget.parentWidget() is not None else None
        if isinstance(layout, QFormLayout):
            label = layout.labelForField(widget)
            if label is not None:
                label.setVisible(visible)

    def _set_compact_profile_field_widths(self) -> None:
        for widget in (
            self.loss_combo,
            self.cc_kernel_edit,
            self.winsorize_lower_edit,
            self.winsorize_upper_edit,
            self.moments_scale_edit,
            self.moments_order_edit,
            self.moments_orientation_combo,
            self.moments_scaling_check,
            self.rigid_loss_combo,
            self.rigid_mi_bins_edit,
            self.rigid_scales_edit,
            self.rigid_iterations_edit,
            self.rigid_lr_edit,
            self.affine_loss_combo,
            self.affine_mi_bins_edit,
            self.affine_scales_edit,
            self.affine_iterations_edit,
            self.affine_lr_edit,
            self.greedy_scales_edit,
            self.greedy_iterations_edit,
            self.greedy_lr_edit,
            self.greedy_optimizer_combo,
            self.syn_loss_combo,
            self.syn_cc_kernel_edit,
            self.syn_scales_edit,
            self.syn_iterations_edit,
            self.syn_lr_edit,
            self.syn_optimizer_combo,
            self.smooth_warp_sigma_edit,
            self.smooth_grad_sigma_edit,
            self.preview_max_side_edit,
        ):
            widget.setMinimumWidth(80)

    def _build_preview_area(self) -> QWidget:
        preview = QWidget()
        preview_layout = QVBoxLayout(preview)
        preview_layout.setContentsMargins(8, 0, 0, 0)
        title_row = QHBoxLayout()
        moving_title = QLabel("Moving / live overlay")
        fixed_title = QLabel("Fixed")
        _set_help(moving_title, "Selected moving bridge preview. During registration this can show the live magenta/green overlay.")
        _set_help(fixed_title, "Fixed reference bridge preview. Moving outputs are written in this image space.")
        title_row.addWidget(moving_title)
        title_row.addStretch(1)
        title_row.addWidget(fixed_title)
        preview_layout.addLayout(title_row)

        preview_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.moving_preview = QLabel("No moving bridge selected")
        self.fixed_preview = QLabel("No fixed bridge selected")
        _set_help(self.moving_preview, "Selected moving bridge preview. Add a moving bridge row to see it here.")
        _set_help(self.fixed_preview, "Fixed bridge preview. Choose a fixed bridge to see it here.")
        for label in (self.moving_preview, self.fixed_preview):
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setMinimumSize(220, 220)
            label.setStyleSheet("QLabel { background: #101418; color: #d9e2ec; border: 1px solid #323b44; }")
            label.setScaledContents(False)
        preview_splitter.addWidget(self.moving_preview)
        preview_splitter.addWidget(self.fixed_preview)
        preview_splitter.setSizes([1, 1])
        preview_layout.addWidget(preview_splitter, 1)
        return preview

    def _restore_settings(self) -> None:
        fixed = self.settings_store.value("fixed_bridge", "")
        output = self.settings_store.value("output_dir", "")
        if fixed:
            self.fixed_path_edit.setText(str(fixed))
        if output:
            self.output_path_edit.setText(str(output))
        try:
            self._restore_profile_settings()
        except ValueError:
            self.profile_combo.setCurrentText(REGISTRATION_PROFILE_NAMES[0])
            self._apply_settings_to_fields(registration_settings_for_profile(REGISTRATION_PROFILE_NAMES[0]))
        self.update_preview()

    def _restore_profile_settings(self) -> None:
        profile_name = str(self.settings_store.value("profile/name", REGISTRATION_PROFILE_NAMES[0]))
        if profile_name not in REGISTRATION_PROFILE_NAMES:
            profile_name = REGISTRATION_PROFILE_NAMES[0]
        settings = registration_settings_for_profile(profile_name)
        settings.winsorize_enabled = _as_bool(self.settings_store.value("profile/winsorize_enabled", settings.winsorize_enabled))
        settings.winsorize_lower = float(self.settings_store.value("profile/winsorize_lower", settings.winsorize_lower))
        settings.winsorize_upper = float(self.settings_store.value("profile/winsorize_upper", settings.winsorize_upper))
        settings.loss_type = str(self.settings_store.value("profile/loss_type", settings.loss_type))
        settings.cc_kernel_size = int(self.settings_store.value("profile/cc_kernel_size", settings.cc_kernel_size))
        settings.moments_scale = float(self.settings_store.value("profile/moments_scale", settings.moments_scale))
        settings.moments_order = int(self.settings_store.value("profile/moments_order", settings.moments_order))
        settings.moments_orientation = str(self.settings_store.value("profile/moments_orientation", settings.moments_orientation))
        settings.moments_perform_scaling = _as_bool(self.settings_store.value("profile/moments_perform_scaling", settings.moments_perform_scaling))
        settings.rigid_loss_type = str(self.settings_store.value("profile/rigid_loss_type", settings.rigid_loss_type))
        settings.rigid_mi_bins = int(self.settings_store.value("profile/rigid_mi_bins", settings.rigid_mi_bins))
        settings.rigid_scales = parse_float_list(str(self.settings_store.value("profile/rigid_scales", _format_number_list(settings.rigid_scales))), "Rigid scales")
        settings.rigid_iterations = parse_int_list(str(self.settings_store.value("profile/rigid_iterations", _format_number_list(settings.rigid_iterations))), "Rigid iterations")
        settings.rigid_lr = float(self.settings_store.value("profile/rigid_lr", settings.rigid_lr))
        settings.affine_loss_type = str(self.settings_store.value("profile/affine_loss_type", settings.affine_loss_type))
        settings.affine_mi_bins = int(self.settings_store.value("profile/affine_mi_bins", settings.affine_mi_bins))
        settings.affine_scales = parse_float_list(str(self.settings_store.value("profile/affine_scales", _format_number_list(settings.affine_scales))), "Affine scales")
        settings.affine_iterations = parse_int_list(str(self.settings_store.value("profile/affine_iterations", _format_number_list(settings.affine_iterations))), "Affine iterations")
        settings.affine_lr = float(self.settings_store.value("profile/affine_lr", settings.affine_lr))
        settings.greedy_scales = parse_float_list(str(self.settings_store.value("profile/greedy_scales", _format_number_list(settings.greedy_scales))), "Greedy scales")
        settings.greedy_iterations = parse_int_list(str(self.settings_store.value("profile/greedy_iterations", _format_number_list(settings.greedy_iterations))), "Greedy iterations")
        settings.greedy_lr = float(self.settings_store.value("profile/greedy_lr", settings.greedy_lr))
        settings.greedy_optimizer = str(self.settings_store.value("profile/greedy_optimizer", settings.greedy_optimizer))
        settings.greedy_reset = _as_bool(self.settings_store.value("profile/greedy_reset", settings.greedy_reset))
        settings.greedy_offload = _as_bool(self.settings_store.value("profile/greedy_offload", settings.greedy_offload))
        settings.syn_loss_type = str(self.settings_store.value("profile/syn_loss_type", settings.syn_loss_type))
        settings.syn_cc_kernel_size = int(self.settings_store.value("profile/syn_cc_kernel_size", settings.syn_cc_kernel_size))
        settings.syn_scales = parse_float_list(str(self.settings_store.value("profile/syn_scales", _format_number_list(settings.syn_scales))), "SyN scales")
        settings.syn_iterations = parse_int_list(str(self.settings_store.value("profile/syn_iterations", _format_number_list(settings.syn_iterations))), "SyN iterations")
        settings.syn_lr = float(self.settings_store.value("profile/syn_lr", settings.syn_lr))
        settings.syn_optimizer = str(self.settings_store.value("profile/syn_optimizer", settings.syn_optimizer))
        settings.smooth_warp_sigma = float(self.settings_store.value("profile/smooth_warp_sigma", settings.smooth_warp_sigma))
        settings.smooth_grad_sigma = float(self.settings_store.value("profile/smooth_grad_sigma", settings.smooth_grad_sigma))
        settings.preview_enabled = _as_bool(self.settings_store.value("profile/preview_enabled", settings.preview_enabled))
        settings.preview_max_side = int(self.settings_store.value("profile/preview_max_side", settings.preview_max_side))
        validate_registration_settings(settings)
        self._loading_profile_fields = True
        self.profile_combo.setCurrentText(profile_name)
        self._apply_settings_to_fields(settings)
        self._loading_profile_fields = False

    def apply_profile_preset(self, profile_name: str) -> None:
        if self._loading_profile_fields:
            return
        settings = registration_settings_for_profile(profile_name)
        self._apply_settings_to_fields(settings)

    def show_profile_tab(self) -> None:
        self.sidebar_tabs.setCurrentIndex(1)

    def _apply_settings_to_fields(self, settings: RegistrationSettings) -> None:
        ants_pipeline = settings.pipeline == PIPELINE_ANTS_RIGID_AFFINE_SYN
        self.loss_group.setVisible(not ants_pipeline)
        self.preprocess_group.setVisible(ants_pipeline)
        self.moments_group.setVisible(not ants_pipeline)
        self.rigid_group.setVisible(ants_pipeline)
        self.greedy_group.setVisible(not ants_pipeline)
        self.syn_group.setVisible(ants_pipeline)
        self._set_form_row_visible(self.affine_loss_combo, ants_pipeline)
        self._set_form_row_visible(self.affine_mi_bins_edit, ants_pipeline)
        self.loss_combo.setCurrentText(settings.loss_type)
        self.cc_kernel_edit.setText(str(settings.cc_kernel_size))
        self.winsorize_enabled_check.setChecked(settings.winsorize_enabled)
        self.winsorize_lower_edit.setText(_format_number(settings.winsorize_lower))
        self.winsorize_upper_edit.setText(_format_number(settings.winsorize_upper))
        self.moments_scale_edit.setText(_format_number(settings.moments_scale))
        self.moments_order_edit.setText(str(settings.moments_order))
        self.moments_orientation_combo.setCurrentText(settings.moments_orientation)
        self.moments_scaling_check.setChecked(settings.moments_perform_scaling)
        self.rigid_loss_combo.setCurrentText(settings.rigid_loss_type)
        self.rigid_mi_bins_edit.setText(str(settings.rigid_mi_bins))
        self.rigid_scales_edit.setText(_format_number_list(settings.rigid_scales))
        self.rigid_iterations_edit.setText(_format_number_list(settings.rigid_iterations))
        self.rigid_lr_edit.setText(str(settings.rigid_lr))
        self.affine_loss_combo.setCurrentText(settings.affine_loss_type)
        self.affine_mi_bins_edit.setText(str(settings.affine_mi_bins))
        self.affine_scales_edit.setText(_format_number_list(settings.affine_scales))
        self.affine_iterations_edit.setText(_format_number_list(settings.affine_iterations))
        self.affine_lr_edit.setText(str(settings.affine_lr))
        self.greedy_scales_edit.setText(_format_number_list(settings.greedy_scales))
        self.greedy_iterations_edit.setText(_format_number_list(settings.greedy_iterations))
        self.greedy_lr_edit.setText(str(settings.greedy_lr))
        self.greedy_optimizer_combo.setCurrentText(settings.greedy_optimizer)
        self.greedy_reset_check.setChecked(settings.greedy_reset)
        self.greedy_offload_check.setChecked(settings.greedy_offload)
        self.syn_loss_combo.setCurrentText(settings.syn_loss_type)
        self.syn_cc_kernel_edit.setText(str(settings.syn_cc_kernel_size))
        self.syn_scales_edit.setText(_format_number_list(settings.syn_scales))
        self.syn_iterations_edit.setText(_format_number_list(settings.syn_iterations))
        self.syn_lr_edit.setText(str(settings.syn_lr))
        self.syn_optimizer_combo.setCurrentText(settings.syn_optimizer)
        self.smooth_warp_sigma_edit.setText(_format_number(settings.smooth_warp_sigma))
        self.smooth_grad_sigma_edit.setText(_format_number(settings.smooth_grad_sigma))
        self.preview_enabled_check.setChecked(settings.preview_enabled)
        self.preview_max_side_edit.setText(str(settings.preview_max_side))

    def _settings_from_profile_fields(self) -> RegistrationSettings:
        settings = registration_settings_for_profile(self.profile_combo.currentText())
        settings.loss_type = self.loss_combo.currentText()
        settings.cc_kernel_size = int(self.cc_kernel_edit.text())
        settings.winsorize_enabled = self.winsorize_enabled_check.isChecked()
        settings.winsorize_lower = float(self.winsorize_lower_edit.text())
        settings.winsorize_upper = float(self.winsorize_upper_edit.text())
        settings.moments_scale = float(self.moments_scale_edit.text())
        settings.moments_order = int(self.moments_order_edit.text())
        settings.moments_orientation = self.moments_orientation_combo.currentText()
        settings.moments_perform_scaling = self.moments_scaling_check.isChecked()
        settings.rigid_loss_type = self.rigid_loss_combo.currentText()
        settings.rigid_mi_bins = int(self.rigid_mi_bins_edit.text())
        settings.rigid_scales = parse_float_list(self.rigid_scales_edit.text(), "Rigid scales")
        settings.rigid_iterations = parse_int_list(self.rigid_iterations_edit.text(), "Rigid iterations")
        settings.rigid_lr = float(self.rigid_lr_edit.text())
        settings.affine_loss_type = self.affine_loss_combo.currentText()
        settings.affine_mi_bins = int(self.affine_mi_bins_edit.text())
        settings.affine_scales = parse_float_list(self.affine_scales_edit.text(), "Affine scales")
        settings.affine_iterations = parse_int_list(self.affine_iterations_edit.text(), "Affine iterations")
        settings.affine_lr = float(self.affine_lr_edit.text())
        settings.greedy_scales = parse_float_list(self.greedy_scales_edit.text(), "Greedy scales")
        settings.greedy_iterations = parse_int_list(self.greedy_iterations_edit.text(), "Greedy iterations")
        settings.greedy_lr = float(self.greedy_lr_edit.text())
        settings.greedy_optimizer = self.greedy_optimizer_combo.currentText()
        settings.greedy_reset = self.greedy_reset_check.isChecked()
        settings.greedy_offload = self.greedy_offload_check.isChecked()
        settings.syn_loss_type = self.syn_loss_combo.currentText()
        settings.syn_cc_kernel_size = int(self.syn_cc_kernel_edit.text())
        settings.syn_scales = parse_float_list(self.syn_scales_edit.text(), "SyN scales")
        settings.syn_iterations = parse_int_list(self.syn_iterations_edit.text(), "SyN iterations")
        settings.syn_lr = float(self.syn_lr_edit.text())
        settings.syn_optimizer = self.syn_optimizer_combo.currentText()
        settings.smooth_warp_sigma = float(self.smooth_warp_sigma_edit.text())
        settings.smooth_grad_sigma = float(self.smooth_grad_sigma_edit.text())
        settings.preview_enabled = self.preview_enabled_check.isChecked()
        settings.preview_max_side = int(self.preview_max_side_edit.text())
        validate_registration_settings(settings)
        return settings

    def _save_profile_settings(self, settings: RegistrationSettings) -> None:
        self.settings_store.setValue("profile/name", self.profile_combo.currentText())
        self.settings_store.setValue("profile/winsorize_enabled", settings.winsorize_enabled)
        self.settings_store.setValue("profile/winsorize_lower", settings.winsorize_lower)
        self.settings_store.setValue("profile/winsorize_upper", settings.winsorize_upper)
        self.settings_store.setValue("profile/loss_type", settings.loss_type)
        self.settings_store.setValue("profile/cc_kernel_size", settings.cc_kernel_size)
        self.settings_store.setValue("profile/moments_scale", settings.moments_scale)
        self.settings_store.setValue("profile/moments_order", settings.moments_order)
        self.settings_store.setValue("profile/moments_orientation", settings.moments_orientation)
        self.settings_store.setValue("profile/moments_perform_scaling", settings.moments_perform_scaling)
        self.settings_store.setValue("profile/rigid_loss_type", settings.rigid_loss_type)
        self.settings_store.setValue("profile/rigid_mi_bins", settings.rigid_mi_bins)
        self.settings_store.setValue("profile/rigid_scales", _format_number_list(settings.rigid_scales))
        self.settings_store.setValue("profile/rigid_iterations", _format_number_list(settings.rigid_iterations))
        self.settings_store.setValue("profile/rigid_lr", settings.rigid_lr)
        self.settings_store.setValue("profile/affine_loss_type", settings.affine_loss_type)
        self.settings_store.setValue("profile/affine_mi_bins", settings.affine_mi_bins)
        self.settings_store.setValue("profile/affine_scales", _format_number_list(settings.affine_scales))
        self.settings_store.setValue("profile/affine_iterations", _format_number_list(settings.affine_iterations))
        self.settings_store.setValue("profile/affine_lr", settings.affine_lr)
        self.settings_store.setValue("profile/greedy_scales", _format_number_list(settings.greedy_scales))
        self.settings_store.setValue("profile/greedy_iterations", _format_number_list(settings.greedy_iterations))
        self.settings_store.setValue("profile/greedy_lr", settings.greedy_lr)
        self.settings_store.setValue("profile/greedy_optimizer", settings.greedy_optimizer)
        self.settings_store.setValue("profile/greedy_reset", settings.greedy_reset)
        self.settings_store.setValue("profile/greedy_offload", settings.greedy_offload)
        self.settings_store.setValue("profile/syn_loss_type", settings.syn_loss_type)
        self.settings_store.setValue("profile/syn_cc_kernel_size", settings.syn_cc_kernel_size)
        self.settings_store.setValue("profile/syn_scales", _format_number_list(settings.syn_scales))
        self.settings_store.setValue("profile/syn_iterations", _format_number_list(settings.syn_iterations))
        self.settings_store.setValue("profile/syn_lr", settings.syn_lr)
        self.settings_store.setValue("profile/syn_optimizer", settings.syn_optimizer)
        self.settings_store.setValue("profile/smooth_warp_sigma", settings.smooth_warp_sigma)
        self.settings_store.setValue("profile/smooth_grad_sigma", settings.smooth_grad_sigma)
        self.settings_store.setValue("profile/preview_enabled", settings.preview_enabled)
        self.settings_store.setValue("profile/preview_max_side", settings.preview_max_side)

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
            if self.jobs[row].live_preview is not None:
                self.moving_preview.setPixmap(self.jobs[row].live_preview)
                self.moving_preview.setToolTip(f"Live overlay: {self.jobs[row].moving_bridge}")
            else:
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
        try:
            settings = self._settings_from_profile_fields()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid registration profile", str(exc))
            return
        settings.device = self.device_edit.text().strip() or settings.device
        settings.progress_bar = False
        run_output_dir = make_run_output_dir(Path(self.output_path_edit.text()))
        settings.metrics_dir = run_output_dir / "_metrics"
        self._save_profile_settings(settings)
        for job in self.jobs:
            job.live_preview = None
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
        if name == "preview_update" and row_index >= 0 and row_index < len(self.jobs):
            preview = event.get("preview")
            if isinstance(preview, np.ndarray):
                self.jobs[row_index].live_preview = rgb_preview_pixmap(preview, max_side=420)
                if self.job_list.currentRow() != row_index:
                    self.job_list.setCurrentRow(row_index)
                else:
                    self.update_preview()
            message = self._status_message(event)
            if message:
                self.statusBar().showMessage(message)
            return

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
        if name == "preview_update":
            stage = event.get("stage", "Registration")
            scale = event.get("scale")
            scale_text = f" scale {scale}" if scale is not None else ""
            return f"{prefix}Updated {stage}{scale_text} overlay"
        if name == "preview_failed":
            stage = event.get("stage", "Registration")
            return f"{prefix}{stage} overlay unavailable: {event.get('error')}"
        if name == "batch_start" and event.get("output_dir"):
            return f"Running in {event.get('output_dir')}"
        if name == "batch_complete":
            return "Batch complete"
        return ""

    def _set_running(self, running: bool) -> None:
        self.add_fixed_action.setEnabled(not running)
        self.add_moving_action.setEnabled(not running)
        self.add_payload_action.setEnabled(not running)
        self.attach_payload_button.setEnabled(not running)
        self.remove_action.setEnabled(not running)
        self.start_action.setEnabled(not running)
        self.stop_action.setEnabled(running)
        for widget in self.profile_widgets:
            widget.setEnabled(not running)
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


def rgb_preview_pixmap(array: np.ndarray, max_side: int = 420) -> QPixmap:
    array = np.ascontiguousarray(array, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Unsupported RGB preview dimensions: {array.shape}")
    h, w, _ = array.shape
    qimage = QImage(array.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
    pixmap = QPixmap.fromImage(qimage)
    return pixmap.scaled(max_side, max_side, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)


def _format_number_list(values: List[float]) -> str:
    return ", ".join(_format_number(value) for value in values)


def _format_number(value: float) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return str(value)


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "on"}


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
