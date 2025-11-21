# wrappers/keras_nn.py

from typing import Tuple

from tensorflow import keras
from scikeras.wrappers import KerasRegressor


def keras_nn(
    hidden_layer_sizes: Tuple[int, ...] = (128, 64),
    learning_rate: float = 1e-3,
    batch_size: int = 256,
    epochs: int = 50,
) -> KerasRegressor:
    """
    SciKeras-KerasRegressor, der in eine sklearn-Pipeline passt.

    - build_model(meta): baut NUR das Modell (ohne compile)
    - Compile findet ausschließlich über KerasRegressor-Argumente statt.
    """

    def build_model(meta):
        # Anzahl Features nach Preprocessing+FS
        n_features = meta["n_features_in_"]

        # Hyperparameter (können via GridSearch überschrieben werden)
        hl_sizes = meta.get("hidden_layer_sizes", hidden_layer_sizes)

        inputs = keras.Input(shape=(n_features,))
        x = inputs
        for units in hl_sizes:
            x = keras.layers.Dense(units, activation="relu")(x)
        outputs = keras.layers.Dense(1)(x)  # Regression

        model = keras.Model(inputs=inputs, outputs=outputs)
        return model

    reg = KerasRegressor(
        model=build_model,
        # Build-Parameter, die in meta landen:
        hidden_layer_sizes=hidden_layer_sizes,
        # Compile-Parameter:
        optimizer=keras.optimizers.Adam,
        optimizer__learning_rate=learning_rate,
        loss="mse",
        # Fit-Parameter:
        batch_size=batch_size,
        epochs=epochs,
        verbose=0,
    )

    return reg
