import logging

import tensorflow as tf
from PyQt5.QtCore import QThread, pyqtSignal
from tensorflow.keras.models import load_model
from tensorflow_addons.layers import GroupNormalization

from Model.custom_layers import DropoutAttentionBlock, SpatialConcreteDropout


class ModelLoaderThread(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(Exception)

    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path
        logging.info("ModelLoaderThread initialized with configuration.")

    def run(self):
        try:
            logging.info(f"Loading model from {self.model_path}")

            model = load_model(
                self.model_path,
                custom_objects={
                    'DropoutAttentionBlock': DropoutAttentionBlock,
                    'GroupNormalization': GroupNormalization,
                    'SpatialConcreteDropout': SpatialConcreteDropout,
                }
            )
            model = tf.keras.models.Model(inputs=model.input, outputs=[model.layers[-2].output, model.output])

            logging.info("Model loaded successfully.")
            self.finished.emit(model)  # Emit the loaded model
        except Exception as e:
            logging.error(f"Error loading model: {e}", exc_info=True)
            self.error.emit(e)
