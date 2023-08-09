import os
import tensorflow as tf
import align_pdb_to_probes as align
from tqdm.keras import TqdmCallback


def _model():
    x = inputs = tf.keras.Input(shape=(2048,))
    for units in (2048, 1024, 512, 256, 128, 64, 32):
        x = tf.keras.layers.Dense(units, activation="relu")(x)

    sin = tf.keras.layers.Dense(3, name="sin")(x)
    cos = tf.keras.layers.Dense(3, name="cos")(x)
    return tf.keras.Model(inputs=inputs, outputs=[sin, cos])



def train():
    _train_gen = align.generate_dataset("train")
    _test_gen = align.generate_dataset("test")

    # data, *_ = next(_train_gen)

    input = tf.TensorSpec(shape=(2048,), dtype=tf.float32, name="input")
    output_sin = tf.TensorSpec(shape=(3), dtype=tf.float32, name="output_sin")
    output_cos = tf.TensorSpec(shape=(3), dtype=tf.float32, name="output_cos")

    train_dataset = tf.data.Dataset.from_generator(
        lambda: _train_gen, output_signature=(input, (output_sin, output_cos))
    )
    test_dataset = tf.data.Dataset.from_generator(
        lambda: _test_gen, output_signature=(input, (output_sin, output_cos))
    )
    
    print(train_dataset)

    epochs: int = 100
    batch_size: int = 8
    steps_per_epoch: int = 10000
    validation_steps: int = 1000
    name: str = "test_1"

    train_dataset = train_dataset.repeat(epochs).batch(batch_size=batch_size)

    test_dataset = test_dataset.repeat(epochs).batch(batch_size=batch_size)
    model = _model()
    model.summary()

    optimiser = tf.keras.optimizers.Adam(lr=1e-4)
    model.compile(optimizer=optimiser, loss="mse",  metrics=['mse'])

    logger = tf.keras.callbacks.CSVLogger(f"train_{name}.csv", append=True)
    reduce_lr_on_plat = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.8,
        patience=5,
        verbose=1,
        mode="auto",
        cooldown=5,
        min_lr=1e-7,
    )
    weight_path: str = f"models/{name}.best.hdf5"

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        weight_path,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        mode="min",
        save_weights_only=False,
    )
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=f"./logs/{name}", histogram_freq=1, profile_batch=(10, 30)
    )
    
    callbacks_list = [
        checkpoint,
        # reduce_lr_on_plat,
        # TqdmCallback(verbose=2),
        # tensorboard_callback,
    ]

    model.fit(
        train_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=test_dataset,
        validation_steps=validation_steps,
        callbacks=callbacks_list,
        verbose=1,
        use_multiprocessing=True,
    )


if __name__ == "__main__":
    train()
