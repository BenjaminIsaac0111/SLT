import logging
import numpy as np
import tensorflow as tf
from PyQt5.QtCore import QObject, pyqtSignal


class ModelProcessor(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)

    def __init__(self, data_loader, model):
        super().__init__()
        self.point_embs = None
        self.point_pred = None
        self.point_true = None
        self.data_loader = data_loader
        self.model = model
        self.memory_bank = None

    @tf.function(jit_compile=True)
    def xla_predict(self, x):
        embs, preds = self.model(x, training=False)
        return embs, preds

    @staticmethod
    def extract_central_pixels(preds, y, embs):
        preds_center = preds[:, preds.shape[1] // 2, preds.shape[2] // 2, :]
        y_center = y[:, y.shape[1] // 2, y.shape[2] // 2, :]
        emb_center = embs[:, embs.shape[1] // 2, embs.shape[2] // 2, :]
        return preds_center, y_center, emb_center

    def initialize_memory_bank(self, output_channels, embedding_dim):
        self.point_true = np.empty((0, output_channels))
        self.point_pred = np.empty((0, output_channels))
        self.point_embs = np.empty((0, embedding_dim))

    def update_memory_bank(self, preds_center, y_center, emb_center):
        self.point_pred = np.concatenate((self.point_pred, preds_center), axis=0)
        self.point_true = np.concatenate((self.point_true, y_center), axis=0)
        self.point_embs = np.concatenate((self.point_embs, emb_center), axis=0)

    def finalise_memory_bank(self):
        point_pred_argmax = np.argmax(self.point_pred, axis=-1).astype(np.float32)
        point_true_argmax = np.argmax(self.point_true, axis=-1).astype(np.float32)
        point_pred_flat = point_pred_argmax.flatten()[:, np.newaxis]
        point_true_flat = point_true_argmax.flatten()[:, np.newaxis]
        point_embs_flat = self.point_embs.reshape(-1, self.point_embs.shape[-1]).astype(np.float32)
        self.memory_bank = np.concatenate([point_pred_flat, point_true_flat, point_embs_flat], axis=-1)


class MemoryBanker(ModelProcessor):
    memory_bank_updated = pyqtSignal(object)

    def run(self):
        logging.info("Memory Bank Worker started processing")
        ds = iter(self.data_loader)
        output_channels = self.model.layers[-1].output_shape[-1]
        embedding_dim = self.model.layers[-2].output_shape[-1]
        self.initialize_memory_bank(output_channels, embedding_dim)

        for index, (x, y) in enumerate(ds):
            embs, preds = self.xla_predict(x)
            preds_center, y_center, emb_center = self.extract_central_pixels(preds, y, embs)
            self.update_memory_bank(preds_center, y_center, emb_center)
            self.progress.emit(index + 1)

        self.finalize_memory_bank()
        self.memory_bank_updated.emit(self.memory_bank)
        logging.info("Memory Bank Worker finished processing")
        self.finished.emit()


class AnomalyDetector(ModelProcessor):
    results_bank = pyqtSignal(object, list)

    def __init__(self, data_loader, model, meta_model):
        super().__init__(data_loader, model)
        self.meta_model = meta_model
        self.filenames = []  # List to store filenames of detected anomalies
        self.point_anomalies = np.empty((0, 1))

    def run(self):
        logging.info("Starting Inference with Anomaly Detection.")
        ds = iter(self.data_loader)
        output_channels = self.model.layers[-1].output_shape[-1]
        embedding_dim = self.model.layers[-2].output_shape[-1]
        self.initialize_memory_bank(output_channels, embedding_dim)

        for index, (filename, (x, y)) in enumerate(ds):
            embs, preds = self.xla_predict(x)
            preds_center, y_center, emb_center = self.extract_central_pixels(preds, y, embs)
            is_anomaly = self.predict_anomalies(emb_center)
            self.update_memory_bank_with_anomalies(
                preds_center,
                y_center,
                emb_center,
                is_anomaly,
                filename,
            )
            self.progress.emit(index + 1)

        self.finalise_memory_bank_with_anomalies()
        self.results_bank.emit(self.memory_bank, self.filenames)

        # Emit both memory bank and anomaly filenames
        logging.info("Anomaly Detector finished processing and memory bank updated with anomaly detection results")
        self.finished.emit()

    def predict_anomalies(self, emb_center):
        embs_reshaped = tf.reshape(emb_center, [emb_center.shape[0], -1]).numpy()
        if isinstance(self.meta_model, tf.keras.Model):
            anomaly = self.meta_model.predict(embs_reshaped).reshape(-1, 1) >= 0.5
        else:
            anomaly = self.meta_model.predict(embs_reshaped).reshape(-1, 1)
        return anomaly

    def update_memory_bank_with_anomalies(self, preds_center, y_center, emb_center, is_anomaly, filename):
        super().update_memory_bank(preds_center, y_center, emb_center)
        negated_anomaly = np.logical_not(is_anomaly)  # TODO: Not sure why, but I have to do this for the meta model...
        self.point_anomalies = np.concatenate((self.point_anomalies, negated_anomaly), axis=0)

        if isinstance(filename, tf.Tensor):
            filename_str = filename.numpy().item()
        else:
            filename_str = filename

        # Negate the specific boolean value is_anomaly[0][0] before appending
        negated_specific_anomaly = not is_anomaly[0][0]
        self.filenames.append((filename_str, negated_specific_anomaly))

    def finalise_memory_bank_with_anomalies(self):
        point_pred_argmax = np.argmax(self.point_pred, axis=-1).astype(np.float32)
        point_true_argmax = np.argmax(self.point_true, axis=-1).astype(np.float32)
        point_pred_flat = point_pred_argmax.flatten()[:, np.newaxis]
        point_true_flat = point_true_argmax.flatten()[:, np.newaxis]
        point_embs_flat = self.point_embs.reshape(-1, self.point_embs.shape[-1]).astype(np.float32)
        point_anomaly_flat = self.point_anomalies.flatten()[:, np.newaxis]
        self.memory_bank = np.concatenate([point_pred_flat, point_true_flat, point_embs_flat, point_anomaly_flat],
                                          axis=-1)
