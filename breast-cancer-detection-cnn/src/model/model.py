from tensorflow import keras
from keras import layers
from keras import metrics, losses
from tensorflow.keras.applications import EfficientNetB0
import tensorflow as tf

from pathlib import Path


def make_thresholded_metric(metric_class, threshold=0.5, name=None):
    metric = metric_class()
    def thresholded(y_true, y_pred):
        y_pred_binary = tf.cast(y_pred >= threshold, tf.float32)
        return metric(y_true, y_pred_binary)
    thresholded.__name__ = name or f"{metric.name}_thr{threshold}"
    return thresholded


class Model:
    def __init__(self):
        self.neural_network = None
        self.PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
        self.TRAINED_MODELS_DIR = self.PROJECT_ROOT / "trained_models"

    def add_augmentation(self, input):
        input_augmentation = keras.Sequential([
            layers.RandomRotation(0.02),
            layers.RandomZoom(0.1),
            layers.RandomTranslation(0.05, 0.05)
        ])
        return input_augmentation(input)

    def build_split_network(self, input_shape=(512, 512, 2)):
        input_tensor = layers.Input(shape=input_shape)
        augmented_layer = self.add_augmentation(input_tensor)

        cc_channel = layers.Lambda(lambda t: t[..., 0:1])(augmented_layer)
        mlo_channel = layers.Lambda(lambda t: t[..., 1:2])(augmented_layer)

        cc_rgb = tf.tile(cc_channel, [1, 1, 1, 3])
        mlo_rgb = tf.tile(mlo_channel, [1, 1, 1, 3])

        backbone = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(512, 512, 3))
        backbone.trainable = False

        emb_cc = backbone(cc_rgb)
        emb_mlo = backbone(mlo_rgb)

        emb_cc = layers.GlobalAveragePooling2D()(emb_cc)
        emb_mlo = layers.GlobalAveragePooling2D()(emb_mlo)

        fused = layers.Concatenate()([emb_cc, emb_mlo,
                                      layers.Subtract()([emb_cc, emb_mlo]),
                                      layers.Multiply()([emb_cc, emb_mlo])])

        x = layers.Dense(256, activation="relu")(fused)
        x = layers.Dropout(0.2)(x)
        out = layers.Dense(1, activation="sigmoid")(x)

        self.neural_network = keras.Model(input_tensor, out)
        return self.neural_network

    def compile(self, learning_rate=1e-5, threshold=0.3):
        self.neural_network.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss="binary_crossentropy",
            metrics=[
                metrics.AUC(name="auc"),
                make_thresholded_metric(metrics.Recall, threshold=threshold, name="sensitivity"),
                make_thresholded_metric(metrics.Precision, threshold=threshold, name="precision")
            ]
        )

    def train(self, training_data, training_labels, validation_data, validation_label, epochs=50, batch_size=32, class_weight=None):
        return self.neural_network.fit(
            x=training_data,
            y=training_labels,
            validation_data=(validation_data, validation_label),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight
        )

    def evaluate(self, data, labels):
        results = self.neural_network.evaluate(data, labels, verbose=2, batch_size=1, return_dict=True)
        print(results)
        return results

    def save_model(self, model_name: str) -> str:
        base = self.TRAINED_MODELS_DIR
        base.mkdir(parents=True, exist_ok=True)

        name = model_name
        out_dir = base / name
        counter = 1
        while out_dir.exists():
            name = f"{model_name}_{counter}"
            out_dir = base / name
            counter += 1

        out_dir.mkdir(parents=True, exist_ok=False)
        out_path = out_dir / f"{name}.keras"

        self.neural_network.save(str(out_path))
        return str(out_path)

    def load_model(self, path):
        return keras.models.load_model(path)

    def specificity(y_true, y_pred):
        y_pred = tf.round(y_pred)
        y_true = tf.cast(y_true, tf.float32)
        tn = tf.reduce_sum((1 - y_true) * (1 - y_pred))
        fp = tf.reduce_sum((1 - y_true) * y_pred)
        return tn / (tn + fp + tf.keras.backend.epsilon())
