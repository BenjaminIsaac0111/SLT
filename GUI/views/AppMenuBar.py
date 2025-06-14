from __future__ import annotations

"""Menu bar for the Smart Annotation GUI."""

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QAction, QActionGroup, QMenuBar, QFileDialog

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
