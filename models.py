"""Model builders: an MLP baseline and the CNN used for the final system."""
from tensorflow import keras
from tensorflow.keras import layers


def build_mlp() -> keras.Sequential:
    """Module 1 baseline: Flatten -> Dense(256) -> Dense(128) -> Dense(26, softmax)."""
    model = keras.Sequential([
        keras.Input(shape=(28, 28, 1)),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(26, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def _activation_layer(activation: str):
    """Return an activation layer; LeakyReLU/ELU need a layer, not a string (module 2)."""
    if activation == "leaky_relu":
        return layers.LeakyReLU(negative_slope=0.1)
    if activation == "elu":
        return layers.ELU()
    return layers.Activation("relu")


def build_cnn(
    activation: str = "relu",
    use_bn: bool = True,
    dropout: float = 0.4,
    pool: str = "max",
    padding: str = "same",
    optimizer="adam",
) -> keras.Sequential:
    """Module 5 CNN: [Conv -> BN -> activation -> Pool] x2 -> Dropout -> Dense -> softmax."""
    pool_layer = layers.MaxPooling2D if pool == "max" else layers.AveragePooling2D

    model_layers = [keras.Input(shape=(28, 28, 1))]

    model_layers.append(layers.Conv2D(32, 3, strides=1, padding=padding))
    if use_bn:
        model_layers.append(layers.BatchNormalization())  # Module 4: normalise before activation
    model_layers.append(_activation_layer(activation))
    model_layers.append(pool_layer(pool_size=2, strides=2))

    model_layers.append(layers.Conv2D(64, 3, strides=1, padding=padding))
    if use_bn:
        model_layers.append(layers.BatchNormalization())
    model_layers.append(_activation_layer(activation))
    model_layers.append(pool_layer(pool_size=2, strides=2))

    model_layers.append(layers.Flatten())
    if dropout > 0:
        model_layers.append(layers.Dropout(dropout))  # Module 4: reduces overfitting
    model_layers.append(layers.Dense(128, activation="relu"))
    model_layers.append(layers.Dense(26, activation="softmax"))

    model = keras.Sequential(model_layers)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model
