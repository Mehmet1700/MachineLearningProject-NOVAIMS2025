# wrappers/keras_nn.py

from typing import Tuple

from tensorflow import keras
from scikeras.wrappers import KerasRegressor


def keras_nn(
    hidden_layer_sizes: Tuple[int, ...] = (128, 64),
    learning_rate: float = 1e-3,
    dropout_rate: float | None = None,
    use_batchnorm: bool = False,
    batch_size: int = 256,
    epochs: int = 50,
    patience: int = 10,
    validation_split: float = 0.1,
) -> KerasRegressor:
    """SciKeras-KerasRegressor, der in eine sklearn-Pipeline passt."""

    def build_model(
        meta,
        hidden_layer_sizes=hidden_layer_sizes,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        use_batchnorm=use_batchnorm,
        **kwargs,
    ):
        """Build a dense NN honoring SciKeras hyperparameters."""
        keras.backend.clear_session()
        n_features = meta["n_features_in_"]

        # SciKeras pushes explicit overrides via kwargs/meta.
        hl_sizes = kwargs.get(
            "hidden_layer_sizes", meta.get("hidden_layer_sizes", hidden_layer_sizes)
        )
        lr = kwargs.get("learning_rate", meta.get("learning_rate", learning_rate))
        dr = kwargs.get("dropout_rate", meta.get("dropout_rate", dropout_rate))
        bn = kwargs.get("use_batchnorm", meta.get("use_batchnorm", use_batchnorm))

        inputs = keras.Input(shape=(n_features,))
        x = inputs
        for idx, units in enumerate(hl_sizes):
            x = keras.layers.Dense(units, name=f"dense_{idx}")(x)
            if bn:
                x = keras.layers.BatchNormalization(name=f"bn_{idx}")(x)
            x = keras.layers.Activation("relu", name=f"relu_{idx}")(x)
            if dr:
                x = keras.layers.Dropout(dr, name=f"dropout_{idx}")(x)
        outputs = keras.layers.Dense(1, name="output")(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        optimizer = keras.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer=optimizer, loss="mse")

        # Some tf.keras versions lack the `compiled` attribute SciKeras expects.
        if not hasattr(model, "compiled"):
            setattr(model, "compiled", True)
        else:
            model.compiled = True
        return model

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=0,
    )

    reg = KerasRegressor(
        model=build_model,
        hidden_layer_sizes=hidden_layer_sizes,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        use_batchnorm=use_batchnorm,
        # Fit-Parameter:
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        callbacks=[early_stop],
        verbose=0,
    )

    return reg
