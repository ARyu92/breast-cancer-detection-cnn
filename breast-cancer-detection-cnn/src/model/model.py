from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path

class Model:
    def __init__(self):
        self.neural_network = None

        # Always set project root relative to this file's location
        self.PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
        self.TRAINED_MODELS_DIR = self.PROJECT_ROOT / "trained_models"


    def build_network(self, layer_input):
        self.neural_network = keras.Sequential()
        self.neural_network.add(layers.Input(shape=(640, 640, 2)))

        for layer in layer_input:
            if layer["layer_type"] == "Conv2D":
                self.neural_network.add(layers.Conv2D(
                    filters=layer["filters"],
                    kernel_size=layer["kernel"],
                    strides=layer.get("stride", 1),
                    activation=layer["activation"],
                    padding="same"
                ))

            elif layer["layer_type"] == "MaxPool":
                self.neural_network.add(layers.MaxPooling2D(
                    pool_size=layer["pool_size"]
                ))

            elif layer["layer_type"] == "Dense":
                self.neural_network.add(layers.Dense(
                    units=layer["units"],
                    activation=layer["activation"],
                    dtype=layer.get("dtype", None)
                ))

            elif layer["layer_type"] == "Dropout":
                self.neural_network.add(layers.Dropout(rate=layer["rate"]))

            elif layer["layer_type"] == "BatchNorm":
                self.neural_network.add(layers.BatchNormalization())

            elif layer["layer_type"] == "Flatten":
                self.neural_network.add(layers.Flatten())

            elif layer["layer_type"] == "GlobalAvgPool":
                self.neural_network.add(layers.GlobalAveragePooling2D())

        return self.neural_network

    def compile(self, learning_rate=1e-4):
        self.neural_network.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

    def train(self, training_data, training_labels, validation_data, validation_label, epochs=5, batch_size=2):
        return self.neural_network.fit(
            x=training_data,
            y=training_labels,
            validation_data=(validation_data, validation_label),
            epochs=epochs,
            batch_size=batch_size
        )

    def evaluate(self, data, labels):
        loss, accuracy = self.neural_network.evaluate(data, labels, verbose=2, batch_size=1)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")

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
