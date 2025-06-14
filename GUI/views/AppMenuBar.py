"""Menu bar for the Smart Annotation GUI."""

from __future__ import annotations

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QAction,
    QActionGroup,
    QMenuBar,
    QFileDialog,
    QInputDialog,
)

from GUI.configuration.configuration import PROJECT_EXT


class AppMenuBar(QMenuBar):
    """Menu bar emitting semantic application-level signals."""

    # ---------------- high-level intents -----------------------------
    request_load_project = pyqtSignal(str)
    request_save_project = pyqtSignal()
    request_save_project_as = pyqtSignal(str)
    request_export_annotations = pyqtSignal(str)
    request_generate_annos = pyqtSignal()
    request_set_ann_method = pyqtSignal(str)
    request_set_nav_policy = pyqtSignal(str)
    request_annotation_preview = pyqtSignal()
    request_create_folds = pyqtSignal(str, str, int)
    request_build_hdf5 = pyqtSignal(str, str, str, str, int, int)

    # ----------------------------------------------------------------
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._build_file_menu()
        self._build_actions_menu()

    # ----------------------------------------------------------------
    def _build_file_menu(self) -> None:
        """Create the File menu."""
        file_menu = self.addMenu("&File")

        act_load = file_menu.addAction("Load Project…")
        act_load.triggered.connect(self._pick_project_to_load)

        act_save = file_menu.addAction("Save")
        act_save.setShortcut("Ctrl+S")
        act_save.triggered.connect(self.request_save_project)

        act_save_as = file_menu.addAction("Save As…")
        act_save_as.triggered.connect(self._pick_path_to_save_as)

    # ----------------------------------------------------------------
    def _build_actions_menu(self) -> None:
        """Create the Actions menu and its submenus."""
        actions_menu = self.addMenu("&Actions")

        ann_menu = actions_menu.addMenu("Annotations")
        act_gen = ann_menu.addAction("Generate Annotations…")
        act_gen.triggered.connect(self.request_generate_annos)
        ann_menu.addSeparator()
        act_preview = ann_menu.addAction("Preview Annotation Overlays…")
        act_preview.triggered.connect(self.request_annotation_preview)
        act_export = ann_menu.addAction("Export Annotations…")
        act_export.triggered.connect(self._pick_path_to_export)

        act_cv = actions_menu.addAction("Create CV Folds…")
        act_cv.triggered.connect(self._pick_folds_dirs)
        act_h5 = actions_menu.addAction("Build Training HDF5…")
        act_h5.triggered.connect(self._pick_hdf5_args)

        # ----- Annotation Method ------------------------------------
        method_menu = actions_menu.addMenu("Annotation Method")
        grp = QActionGroup(self)
        for label in [
            "Local Uncertainty Maxima",
            "Equidistant Spots",
            "Image Centre",
        ]:
            act = QAction(label, self, checkable=True)
            grp.addAction(act)
            method_menu.addAction(act)
        grp.actions()[0].setChecked(True)
        grp.triggered.connect(lambda a: self.request_set_ann_method.emit(a.text()))

        # -------- Navigation Policy sub-menu -------------------------
        nav_menu = actions_menu.addMenu("Navigation Policy")
        nav_grp = QActionGroup(self)
        for label, name in [
            ("Greedy", "greedy"),
            ("Sequential", "sequential"),
            ("Random", "random"),
        ]:
            act = QAction(label, self, checkable=True)
            act.setData(name)
            nav_grp.addAction(act)
            nav_menu.addAction(act)
        nav_grp.actions()[0].setChecked(True)
        nav_grp.triggered.connect(
            lambda a: self.request_set_nav_policy.emit(a.data())
        )

    # ----------------------------------------------------------------
    def set_checked_annotation_method(self, label: str) -> None:
        """Tick QAction matching *label* and untick others."""
        for act in self.findChildren(QAction):
            if act.text() == label:
                act.setChecked(True)
            elif act.isCheckable():
                act.setChecked(False)

    # ----------------------------------------------------------------
    #  QFileDialog helpers (private)
    # ----------------------------------------------------------------
    def _pick_project_to_load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Project", "", f"Smart-Label Project (*{PROJECT_EXT});;All Files (*)"
        )
        if path:
            self.request_load_project.emit(path)

    def _pick_path_to_save_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Project As", "", f"Smart-Label Project (*{PROJECT_EXT});;All Files (*)"
        )
        if path and not path.endswith(PROJECT_EXT):
            path += PROJECT_EXT
        if path:
            self.request_save_project_as.emit(path)

    def _pick_path_to_export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Annotations", "", "JSON files (*.json);;All files (*)"
        )
        if path:
            self.request_export_annotations.emit(path)

    def _pick_folds_dirs(self) -> None:
        data_dir = QFileDialog.getExistingDirectory(
            self, "Select Image Directory", ""
        )
        if not data_dir:
            return
        out_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", ""
        )
        if not out_dir:
            return

        splits, ok = QInputDialog.getInt(
            self,
            "Cross Validation",
            "Number of folds:",
            3,
            2,
            20,
        )
        if ok:
            self.request_create_folds.emit(data_dir, out_dir, splits)

    def _pick_hdf5_args(self) -> None:
        data_dir = QFileDialog.getExistingDirectory(
            self, "Select Image Directory", ""
        )
        if not data_dir:
            return
        csv_file, _ = QFileDialog.getOpenFileName(
            self, "Select Training CSV", "", "CSV files (*.csv);;All files (*)"
        )
        if not csv_file:
            return
        model_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Segmentation Model",
            "",
            "HDF5 files (*.h5 *.hdf5);;All files (*)",
        )
        if not model_path:
            return
        out_file, _ = QFileDialog.getSaveFileName(
            self, "Output HDF5 File", "", "HDF5 files (*.h5 *.hdf5);;All files (*)"
        )
        if not out_file:
            return
        size, ok = QInputDialog.getInt(
            self,
            "Subsample",
            "Number of samples (0 = all):",
            0,
            0,
            10_000_000,
        )
        if not ok:
            return

        mc_iter, ok = QInputDialog.getInt(
            self,
            "MC Iterations",
            "Number of stochastic passes:",
            1,
            1,
            512,
        )
        if ok:
            self.request_build_hdf5.emit(
                data_dir, csv_file, model_path, out_file, size, mc_iter
            )
