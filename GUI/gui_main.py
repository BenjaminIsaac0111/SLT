import sys
import uuid
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QDockWidget, QFileDialog, QTabWidget, QDesktopWidget
from GUI.patch_image_viewer import PatchImageViewer


class MainWindow(QMainWindow):
    """
    MainWindow class represents the main window of the Labeling Assistant application.

    This class inherits from QMainWindow and provides functionality to adjust the window size,
    manage dock widgets, create menus, and validate configuration settings.
    """

    def __init__(self):
        """
        Initializes the MainWindow class.

        Sets the window title, adjusts the window size, configures dock options,
        and creates the menu.
        """
        super().__init__()
        self.config_editor = None
        self.setWindowTitle("Labeling Assistant")
        self.adjustWindowSize()
        self.setDockOptions(self.GroupedDragging | self.AllowTabbedDocks | self.AllowNestedDocks)
        self.setTabPosition(Qt.AllDockWidgetAreas, QTabWidget.North)
        self.dockWidgets = {}
        self.createMenu()

    def adjustWindowSize(self):
        """
        Adjusts the size of the main window to 50% of the screen size and centres it.

        This method calculates a suitable size for the window based on the screen resolution
        and centres the window on the available screen space.
        """
        screen = QApplication.screens()[0]
        screen_size = screen.size()

        window_width = screen_size.width() * 0.5
        window_height = screen_size.height() * 0.5
        self.resize(int(window_width), int(window_height))

        self.move(QDesktopWidget().availableGeometry().center() - self.frameGeometry().center())

    def createMenu(self):
        """
        Creates the main menu for the application.

        This method initializes the menu bar by creating the file menu.
        """
        self.createFileMenu()

    def createFileMenu(self):
        """
        Creates the 'File' menu and adds actions to it.

        This method adds an 'Open Patches Directory' action to the file menu
        and connects it to the openPatchDirectory method.
        """
        fileMenu = self.menuBar().addMenu('File')
        openPatchAction = QAction('Open Patches Directory', self)
        openPatchAction.triggered.connect(self.openPatchDirectory)
        fileMenu.addAction(openPatchAction)

    def openPatchDirectory(self):
        """
        Opens a directory containing image patches.

        This method triggers the creation of a dock widget to display image patches.
        """
        self.createPatchImageViewerDock()

    def createPatchImageViewerDock(self, directory=''):
        """
        Creates and displays a dock widget for viewing image patches.

        Args:
            directory (str): The directory containing the image patches. Default is an empty string.

        A unique ID is generated for each dock widget. The widget is then added to the main window
        and stored in the dockWidgets dictionary.
        """
        unique_id = str(uuid.uuid4())
        patch_viewer = PatchImageViewer(patches_dir=directory)
        patch_viewer_dock = QDockWidget(f"Patch Viewer - {unique_id}", self)
        patch_viewer_dock.setWidget(patch_viewer)
        patch_viewer_dock.setAllowedAreas(Qt.AllDockWidgetAreas)
        patch_viewer_dock.setObjectName(unique_id)
        patch_viewer_dock.destroyed.connect(lambda: self.onDockWidgetClosed(unique_id))
        self.addDockWidget(Qt.LeftDockWidgetArea, patch_viewer_dock)
        patch_viewer_dock.show()
        self.dockWidgets[unique_id] = patch_viewer_dock

    def onDockWidgetClosed(self, unique_id):
        """
        Handles the cleanup process when a dock widget is closed.

        Args:
            unique_id (str): The unique identifier of the dock widget to be removed.

        This method removes the reference to the closed dock widget from the dockWidgets dictionary.
        """
        if unique_id in self.dockWidgets:
            del self.dockWidgets[unique_id]

    @staticmethod
    def validate_config(cfg, required_keys):
        """
        Validates a configuration dictionary against a list of required keys.

        Args:
            cfg (dict): Configuration dictionary to validate.
            required_keys (list): List of keys that are required in the cfg.

        Raises:
            ValueError: If any required keys are missing or empty.
            FileNotFoundError: If file paths in cfg do not exist.
        """
        missing_keys = [key for key in required_keys if key not in cfg or not cfg[key]]
        if missing_keys:
            raise ValueError(f"Missing or empty configuration for keys: {', '.join(missing_keys)}")


if __name__ == '__main__':
    """
    Entry point for the application.

    Creates an instance of QApplication, initializes MainWindow, and starts the
    application event loop.
    """
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())
