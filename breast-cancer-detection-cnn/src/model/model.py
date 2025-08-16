from tensorflow import keras
from keras import layers
from keras import metrics, losses

from pathlib import Path

class Model:
    def __init__(self):
        self.neural_network = None

        # Always set project root relative to this file's location
        self.PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
        self.TRAINED_MODELS_DIR = self.PROJECT_ROOT / "trained_models"


    def build_network(self, layer_input):
        self.neural_network = keras.Sequential()
        self.neural_network.add(layers.Input(shape=(512, 512, 2)))

        # in Model.build_network(), right after Input(...)
        self.neural_network.add(layers.RandomFlip("horizontal"))
        self.neural_network.add(layers.RandomRotation(0.03))
        self.neural_network.add(layers.RandomZoom(0.05))
        self.neural_network.add(layers.RandomTranslation(0.03, 0.03))
        self.neural_network.add(layers.RandomContrast(0.05))
        self.neural_network.add(layers.GaussianNoise(0.02))


        pending_activation = None  # remember the activation to apply after BN

        for layer in layer_input:
            if layer["layer_type"] == "Conv2D":
                pending_activation = layer.get("activation", None)
                self.neural_network.add(layers.Conv2D(
                    filters=layer["filters"],
                    kernel_size=layer["kernel"],
                    strides=layer.get("stride", 1),
                    activation=None,                          # defer
                    padding=layer.get("padding", "same"),
                    kernel_regularizer=layer.get("kernel_regularizer", None)
                ))

            elif layer["layer_type"] == "BatchNorm":
                self.neural_network.add(layers.BatchNormalization())
                if pending_activation:
                    self.neural_network.add(layers.Activation(pending_activation))
                    pending_activation = None

            elif layer["layer_type"] == "MaxPool":
                if pending_activation:                       # no BN came; apply act now
                    self.neural_network.add(layers.Activation(pending_activation))
                    pending_activation = None
                self.neural_network.add(layers.MaxPooling2D(pool_size=layer["pool_size"]))

            elif layer["layer_type"] == "Dense":
                pending_activation = layer.get("activation", None)
                self.neural_network.add(layers.Dense(
                    units=layer["units"],
                    activation=None,                          # defer
                    dtype=layer.get("dtype", None),
                    kernel_regularizer=layer.get("kernel_regularizer", None)
                ))

            elif layer["layer_type"] == "Dropout":
                if pending_activation:
                    self.neural_network.add(layers.Activation(pending_activation))
                    pending_activation = None
                self.neural_network.add(layers.Dropout(rate=layer["rate"]))

            elif layer["layer_type"] == "Flatten":
                if pending_activation:
                    self.neural_network.add(layers.Activation(pending_activation))
                    pending_activation = None
                self.neural_network.add(layers.Flatten())

            elif layer["layer_type"] == "GlobalAvgPool":
                if pending_activation:
                    self.neural_network.add(layers.Activation(pending_activation))
                    pending_activation = None
                self.neural_network.add(layers.GlobalAveragePooling2D())

        if pending_activation:
            self.neural_network.add(layers.Activation(pending_activation))

        return self.neural_network




    def compile(self, learning_rate=1e-4):
        self.neural_network.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss=losses.BinaryCrossentropy(label_smoothing=0.05),
        metrics=["accuracy", metrics.AUC(name="auc")]
    )

    def train(self, training_data, training_labels, validation_data, validation_label, epochs=50, batch_size=32):
        return self.neural_network.fit(
            x=training_data,
            y=training_labels,
            validation_data=(validation_data, validation_label),
            epochs=epochs,
            batch_size=batch_size
        )

    def evaluate(self, data, labels):
        results = self.neural_network.evaluate(data, labels, verbose=2, batch_size=1, return_dict = True)
        print (results)
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
