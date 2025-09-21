import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input, metrics, losses
from tensorflow.keras.applications import EfficientNetB0, DenseNet121, MobileNetV2
from tensorflow.keras.models import Model
import h5py
from pathlib import Path


def make_thresholded_metric(metric_class, threshold=0.3, name=None):
    metric = metric_class()
    def thresholded(y_true, y_pred):
        y_pred_binary = tf.cast(y_pred >= threshold, tf.float32)
        return metric(y_true, y_pred_binary)
    thresholded.__name__ = name or f"{metric.name}_thr{threshold}"
    return thresholded

#This class holds the model and the support methods related to the model functionality.
class BreastCancerModel:
    def __init__(self):
        self.neural_network = None
        self.PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
        self.TRAINED_MODELS_DIR = self.PROJECT_ROOT / "trained_models"
        self.TEMP_MODELS_DIR = self.PROJECT_ROOT / "temp"

    #This helper method adds random rotations, zoom, translation augmentation layers to the model.
    def add_augmentation(self, input):
        input_augmentation = keras.Sequential([
            layers.RandomRotation(0.02),
            layers.RandomZoom(0.1),
            layers.RandomTranslation(0.05, 0.05)
        ])
        return input_augmentation(input)
    
    #This model builds out the entire structure of the model.
    def build_split_network(self, input_shape=(512, 512, 2)):
        input_tensor = layers.Input(shape=input_shape)
        augmented_layer = self.add_augmentation(input_tensor)

        cc_channel = layers.Lambda(lambda t: t[..., 0:1])(augmented_layer)
        mlo_channel = layers.Lambda(lambda t: t[..., 1:2])(augmented_layer)
        #Tile the inputs for CC and MLO views into an image of 3 channels, which is needed for transfer learning to the mobileNetV2 layers.
        cc_rgb = tf.tile(cc_channel, [1, 1, 1, 3])
        mlo_rgb = tf.tile(mlo_channel, [1, 1, 1, 3])

        # CC branch
        cc_input = Input(shape=(512, 512, 3), name="cc_input")
        cc_base = MobileNetV2(include_top=False, weights="imagenet", input_tensor=cc_input)
        self.cc_backbone = Model(cc_input, cc_base.output, name="cc_mobilenetv2")

        # MLO branch
        mlo_input = Input(shape=(512, 512, 3), name="mlo_input")
        mlo_base = MobileNetV2(include_top=False, weights="imagenet", input_tensor=mlo_input)
        self.mlo_backbone = Model(mlo_input, mlo_base.output, name="mlo_mobilenetv2")

        #Freeze the initial backbones, as the pretrained weights should not be altered in phase 1.
        self.cc_backbone.trainable = False
        self.mlo_backbone.trainable = False

        emb_cc = self.cc_backbone(cc_rgb)
        emb_mlo = self.mlo_backbone(mlo_rgb)

        #Apply GlobalAveragePooling2D in order to decrease the amount of trainable filters.
        emb_cc = layers.GlobalAveragePooling2D()(emb_cc)
        emb_mlo = layers.GlobalAveragePooling2D()(emb_mlo)

        #Fuse the layers
        fused = layers.Concatenate()([emb_cc, emb_mlo,
                                      layers.Subtract()([emb_cc, emb_mlo]),
                                      layers.Multiply()([emb_cc, emb_mlo])])

        #Decider head.
        x = layers.Dense(256, activation="relu")(fused)
        out = layers.Dense(1, activation="sigmoid", dtype="float32")(x)

        self.neural_network = keras.Model(input_tensor, out)
        return self.neural_network

    #This function compiles the model
    def compile(self, learning_rate=1e-5, threshold=0.20, optimizer = keras.optimizers.Adam):
        self.neural_network.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss="binary_crossentropy",
            metrics=[
                metrics.AUC(name="auc"),
                make_thresholded_metric(metrics.Recall, threshold=threshold, name="sensitivity"),
                make_thresholded_metric(metrics.Precision, threshold=threshold, name="precision")
            ]
        )

    #Performs a training round according to input parameters.
    def train(self, training_data, training_labels, validation_data, validation_label, epochs=50, batch_size=32, class_weight=None):
        return self.neural_network.fit(
            x=training_data,
            y=training_labels,
            validation_data=(validation_data, validation_label),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight
        )

    #Evalutes the model taking in testing data and labels.
    def evaluate(self, data, labels):
        results = self.neural_network.evaluate(data, labels, verbose=2, batch_size=1, return_dict=True)
        
        return results

    #Saves the model to disk with the training mean and std as parameters.
    def save_model(self, model_name, training_mean, training_std):
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
        out_path = str(out_dir / f"{name}.h5")
        self.neural_network.save(out_path)

        #After saving the model, append the training mean and standard deviation so that it can be applied at inference.
        with h5py.File(out_path, "a") as f:
            f.attrs["training_mean"] = training_mean
            f.attrs["training_std"] = training_std
        return str(out_path)
 
    #Load the model with the training mean and std as attributes.
    def load_model(self, path):
        self.neural_network=  keras.models.load_model(path, compile= False)

        #Afterloading the model, grab the training mean and standard deviation so that it can be applied at inference.
        with h5py.File(path, "r") as f:
            training_mean = f.attrs["training_mean"]
            training_std = f.attrs["training_std"]
            return training_mean, training_std