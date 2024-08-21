import sys

from PyQt5.QtGui import QPixmap, QImage, QPainter
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QApplication
from PyQt5.QtCore import QRunnable, pyqtSignal, QObject, QThreadPool, QPoint, Qt, QRectF, QTimer
import tiffslide as tiffslide


class WSIViewer(QGraphicsView):
    def __init__(self, *args):
        super(WSIViewer, self).__init__(*args)

        self.downsample = None
        self.objective_power = None
        self.mpp_y = None
        self.mpp_x = None
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self._dragging = False
        self._last_drag_point = QPoint()

        # Zoom factors
        self._zoomInFactor = 1.25
        self._zoomOutFactor = 1 / self._zoomInFactor
        self._zoom = 0
        self._zoomStep = 1
        self._zoomRange = [0, 20]

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.tiles = {}  # Cache for tiles
        self.tile_size_w = None
        self.tile_size_h = None
        self.current_level = None
        self.level_dim = None
        self.slide = None
        self.threadpool = QThreadPool()

    def loadWSI(self, path):
        try:
            self.slide = tiffslide.OpenSlide(path)
        except Exception as e:
            print(f"Error loading slide: {e}")

        # Set the current level to the lowest resolution
        self.current_level = self.slide.level_count - 1
        print('Current Level:', self.current_level)

        # Parse properties
        properties = self.slide.properties
        self.mpp_x = float(properties['tiffslide.mpp-x'])
        self.mpp_y = float(properties['tiffslide.mpp-y'])
        self.objective_power = float(properties.get('tiffslide.objective-power', 0))

        # Set the scene dimensions based on the current level dimensions
        self.level_dim = self.slide.level_dimensions[self.current_level]
        self.scene.setSceneRect(0, 0, self.level_dim[0], self.level_dim[1])

        # Adjust the viewer based on tile size
        self.tile_size_w = int(properties[f'tiffslide.level[{self.current_level}].tile-width'])
        self.tile_size_h = int(properties[f'tiffslide.level[{self.current_level}].tile-height'])

        # Make sure on load renderer loads the current view properly.
        QTimer.singleShot(0, self.update_view)

    def closeWSI(self):
        if self.slide is not None:
            self.slide.close()
            self.slide = None
        self.tiles.clear()
        self.scene.clear()

        self.downsample = None
        self.tile_size_w = None
        self.tile_size_h = None
        self.current_level = None
        self.level_dim = None
        self.objective_power = None
        self.mpp_y = None
        self.mpp_x = None

        self.update_view()

    def wheelEvent(self, event):
        # Set Anchors
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

        # Save the scene pos
        oldPos = self.mapToScene(event.pos())

        # Determine Zoom Direction
        if event.angleDelta().y() > 0:
            zoomFactor = self._zoomInFactor
            self._zoom += self._zoomStep
        else:
            zoomFactor = self._zoomOutFactor
            self._zoom -= self._zoomStep

        # Clamp zoom and apply if within range
        self._zoom = max(min(self._zoom, self._zoomRange[1]), self._zoomRange[0])
        self.scale(zoomFactor, zoomFactor)

        # Get the new position
        newPos = self.mapToScene(event.pos())

        # Move scene to old position
        delta = newPos - oldPos
        self.translate(delta.x(), delta.y())

        # Update the current level after zooming
        self.update_current_level()

        # Update the view to reflect the new level
        self.update_view()

    def get_expected_downsample_factor(self):
        """
        Calculate the expected downsample factor based on the current zoom level.
        """
        # Base magnification (objective power)
        base_magnification = self.objective_power

        # Current magnification is base magnification times the zoom scale
        current_magnification = base_magnification * (1 + self._zoom * self._zoomStep)

        # The expected downsample factor is the ratio of base magnification to current magnification
        return base_magnification / current_magnification

    def update_current_level(self):
        expected_downsample = self.get_expected_downsample_factor()

        # For zooming in beyond the base resolution, maintain the highest resolution level
        # This check ensures we only look for a closer level when zooming out
        if expected_downsample >= 1:
            closest_level = None
            min_diff = float('inf')
            selected_downsample = None

            # Iterate through all pyramid levels to find the closest level for zooming out
            for level in range(self.slide.level_count):
                downsample = self.slide.level_downsamples[level]
                diff = abs(expected_downsample - downsample)
                if diff < min_diff:
                    min_diff = diff
                    closest_level = level
                    selected_downsample = downsample

            # Update the current level if a closer match is found and it's different from the current level
            if closest_level is not None and closest_level != self.current_level:
                self.current_level = closest_level
                self.level_dim = self.slide.level_dimensions[self.current_level]
                self.scene.setSceneRect(0, 0, self.level_dim[0], self.level_dim[1])
                self.update_tile_size()
                self.update_view()

            # Optionally, add debug information to monitor changes
            print(
                f"Expected Downsample: {expected_downsample}, Selected Level: {closest_level}, Selected Downsample: {selected_downsample}")
        else:
            # When zooming in beyond the highest resolution, adjust the viewer's zoom but not the level
            # This may involve scaling the image within the viewer without changing the pyramid level
            # Since this does not require changing the pyramid level, no level update is needed here
            # You may implement additional viewer scaling logic here if necessary
            print("Zooming in beyond highest resolution; maintaining highest pyramid level.")

    def update_tile_size(self):
        # Update the tile size for the new level
        properties = self.slide.properties
        self.tile_size_w = int(properties[f'tiffslide.level[{self.current_level}].tile-width'])
        self.tile_size_h = int(properties[f'tiffslide.level[{self.current_level}].tile-height'])

    def update_view(self):
        if self.slide:
            self.downsample = self.slide.level_downsamples[self.current_level]
            # Determine the visible tiles based on the current viewport
            visible_tile_keys = self.visible_tiles()

            # Remove items that are no longer visible
            for key in list(self.tiles.keys()):
                if key not in visible_tile_keys:
                    item = self.tiles.pop(key)
                    self.scene.removeItem(item)

            # Load new visible tiles that are not already loaded
            for key in visible_tile_keys:
                if key not in self.tiles:
                    x, y, level, width, height = key
                    self.load_tile(x, y, level, width, height)

    def visible_tiles(self, buffer=2048):
        """
        Calculate visible tiles with an additional buffer area.

        :param buffer: Number of pixels to extend beyond the current view in all directions.
        """
        self.scene.update()
        self.viewport().update()

        # Get the current viewport and expand it by the buffer
        viewport_rect = self.mapToScene(self.viewport().geometry()).boundingRect()
        buffered_rect = viewport_rect.adjusted(-buffer, -buffer, buffer, buffer)

        visible_tile_keys = set()
        level_dim = self.slide.level_dimensions[self.current_level]

        # Check tiles within the buffered area
        for y in range(0, level_dim[1], self.tile_size_h):
            for x in range(0, level_dim[0], self.tile_size_w):
                tile_rect = QRectF(x, y, self.tile_size_w, self.tile_size_h)
                if buffered_rect.intersects(tile_rect):
                    tile_key = (
                        x * self.downsample, y * self.downsample, self.current_level, self.tile_size_w,
                        self.tile_size_h)
                    visible_tile_keys.add(tile_key)
        return visible_tile_keys

    def load_tile(self, x, y, level, width, height):
        worker = TileLoaderWorker(self.slide, x, y, self.current_level, width, height)
        worker.signals.finished.connect(self.display_tile)
        self.threadpool.start(worker)

    def display_tile(self, key, pixmap):
        if key not in self.tiles:
            item = QGraphicsPixmapItem(pixmap)
            item.setPos(key[0] / self.downsample, key[1] / self.downsample)
            self.tiles[key] = item
            self.scene.addItem(item)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._last_drag_point = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
        else:
            super(WSIViewer, self).mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging:
            delta = event.pos() - self._last_drag_point
            self._last_drag_point = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            self.update_view()
        else:
            super(WSIViewer, self).mouseMoveEvent(event)
            self.update_view()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()
        else:
            super(WSIViewer, self).mouseReleaseEvent(event)

    def resizeEvent(self, event):
        super(WSIViewer, self).resizeEvent(event)
        self.update_view()


class TileLoaderWorker(QRunnable):
    def __init__(self, slide, x, y, level, width, height):
        super().__init__()
        self.slide = slide
        self.level = level
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.signals = TileLoaderSignals()

    def run(self):
        try:
            region = self.slide.read_region((self.x, self.y), self.level, (self.width, self.height), as_array=True,
                                            padding=False)
            # Ensure the alpha channel (transparency) is properly handled
            if region.shape[2] == 4:  # Assuming RGBA format (4 channels)
                qimage = QImage(region.data_loader, region.shape[1], region.shape[0], region.strides[0],
                                QImage.Format_RGBA8888)
            else:
                qimage = QImage(region.data_loader, region.shape[1], region.shape[0], region.strides[0], QImage.Format_RGB888)

            pixmap = QPixmap.fromImage(qimage)
            self.signals.finished.emit((self.x, self.y, self.level, self.width, self.height), pixmap)
        except Exception as e:
            print(f"Exception in thread: {e}")


class TileLoaderSignals(QObject):
    finished = pyqtSignal(tuple, QPixmap)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setAttribute(Qt.AA_UseSoftwareOpenGL, False)
    mainWin = WSIViewer()
    mainWin.loadWSI(path=r'C:\Users\wispy\OneDrive - University of Leeds\DATABACKUP\Clasicc\01004\3699.svs')
    mainWin.show()
    sys.exit(app.exec_())
