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


    #Functions to build split input model.
    def create_convolution_block (self, layer, filters, k, strides):
        layer = layers.Conv2D(filters, kernel_size = k, strides = strides)(layer)
        layer = layers.BatchNormalization()(layer)
        layer = layers.Activation("relu")(layer)
        return layer
    
    #Create a 3 block deep "branch" for each image view tensor.
    def create_three_block_branch(self, input_tensor, filter1 = 32, filter2 = 64, filter3 = 128):
        #block 1
        layer = self.create_convolution_block(input_tensor, filter1, 3, 1)
        layer = layers.MaxPool2D(2)(layer)

        #block2
        layer = self.create_convolution_block(layer, filter2, 3, 1)
        layer = layers.MaxPool2D(2)(layer)

        #block3
        layer = self.create_convolution_block(layer, filter3, 3, 1)
        layer = layers.MaxPool2D(2)(layer)

        return layer

    def build_split_network(self, input_shape = (512, 512, 2), filter1 = 32, filter2 = 64, filter3 = 128):
        #Input is the stacked tensor, each image per channel. 
        input_tensor = layers.Input(shape = input_shape)

        #Split the input by channel using lambdas. CC view is in channel 0, and MLO is in channel 1
        cc_channel = layers.Lambda(lambda t: t[..., 0:1])(input_tensor)
        mlo_channel = layers.Lamda(lambda t: t[..., 1:2])(input_tensor)

        #Create the separate branches for each of the channels. 
        cc_branch = self.create_three_block_branch(cc_channel, filter1 = filter1, filter2= filter2, filter3= filter3)
        mlo_branch = self.create_three_block_branch(mlo_channel, filter1 = filter1, filter2= filter2, filter3= filter3)
        
        #Fuse the two branches, but how? Apply some operations to the feature spaces to gain other insights
        #1. Concatenate, to keep the raw data from both views
        concat = layers.Concatenate()([cc_branch, mlo_branch])

        #2. Subtract, to see the differences between the two feature spaces.
        diff = layers.Subtract()([cc_branch, mlo_branch])

        #3. Multiply, to amplify where the feature space is the same
        product = layers.Multiply()([cc_branch, mlo_branch])

        #4. Combine them all so that there is a representation that can be classified.
        combined = layers.concatenate([concat, diff, product])

        combined = self.create_convolution_block(combined, 256, k = 1, strides = 1)

        layer = self.create_convolution_block(combined, 256, k = 3, stides = 2)
        layer = self.create_convolution_block(layer, 256, k = 3, strides = 1)
        layer = layers.GlobalAveragePooling2D()(layer)

        layer = layers.Dense(256, activation= "relu")(layer)
        layer = layers.Dropout(0.5)(layer)

        output = layers.Dense(1, activation = "sigmoid")(layer)

                
        return Model(input_tensor, output)


    def compile(self, learning_rate=1e-4):
        self.neural_network.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss=losses.BinaryCrossentropy(label_smoothing=0.05),
        metrics=["accuracy", metrics.AUC(name="auc")]
    )

    def train(self, training_data, training_labels, validation_data, validation_label, epochs=100, batch_size=32):
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
