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
        # ---- build a SHARED BACKBONE from your existing layer spec (for 1-channel input) ----
        backbone = keras.Sequential(name="shared_backbone")
        backbone.add(layers.Input(shape=(512, 512, 1)))

        pending_activation = None
        for layer in layer_input:
            if layer["layer_type"] == "Conv2D":
                pending_activation = layer.get("activation", None)
                backbone.add(layers.Conv2D(
                    filters=layer["filters"],
                    kernel_size=layer["kernel"],
                    strides=layer.get("stride", 1),
                    activation=None,  # defer until after BN
                    padding=layer.get("padding", "same"),
                    kernel_regularizer=layer.get("kernel_regularizer", None)
                ))

            elif layer["layer_type"] == "BatchNorm":
                backbone.add(layers.BatchNormalization())
                if pending_activation:
                    backbone.add(layers.Activation(pending_activation))
                    pending_activation = None

            elif layer["layer_type"] == "MaxPool":
                if pending_activation:
                    backbone.add(layers.Activation(pending_activation))
                    pending_activation = None
                backbone.add(layers.MaxPooling2D(pool_size=layer["pool_size"]))

            elif layer["layer_type"] == "Dropout":
                if pending_activation:
                    backbone.add(layers.Activation(pending_activation))
                    pending_activation = None
                backbone.add(layers.Dropout(rate=layer["rate"]))

            elif layer["layer_type"] == "Flatten":
                if pending_activation:
                    backbone.add(layers.Activation(pending_activation))
                    pending_activation = None
                backbone.add(layers.Flatten())

            elif layer["layer_type"] == "GlobalAvgPool":
                if pending_activation:
                    backbone.add(layers.Activation(pending_activation))
                    pending_activation = None
                backbone.add(layers.GlobalAveragePooling2D())

        if pending_activation:
            backbone.add(layers.Activation(pending_activation))

        # ---- main model: single 2-channel input -> split -> shared backbone -> late fusion ----
        inp = layers.Input(shape=(512, 512, 2), name="cc_mlo_stacked")

        cc  = layers.Lambda(lambda t: t[..., 0:1], name="split_cc")(inp)   # (H,W,1)
        mlo = layers.Lambda(lambda t: t[..., 1:2], name="split_mlo")(inp)  # (H,W,1)

        emb_cc  = backbone(cc)
        emb_mlo = backbone(mlo)

        # Late fusion: concat + difference + product
        fused = layers.Concatenate(name="fuse_concat")([
            emb_cc,
            emb_mlo,
            layers.Subtract(name="fuse_subtract")([emb_cc, emb_mlo]),
            layers.Multiply(name="fuse_multiply")([emb_cc, emb_mlo]),
        ])

        # Small classification head
        x = layers.Dense(256, activation="relu", name="head_dense")(fused)
        x = layers.Dropout(0.5, name="head_dropout")(x)
        out = layers.Dense(1, activation="sigmoid", name="pred")(x)

        self.neural_network = keras.Model(inp, out, name="late_fusion_split_input")
        return self.neural_network





    def compile(self, learning_rate=1e-4):
        self.neural_network.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
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
