import argparse
import os
from datetime import datetime

from data import create_pipeline_performance
from model import unet

from tensorflow import keras


def model_builder():
    """Build a model by calling custom_model function.

    The number of output neurons is automatically set to the number of classes
    that make up the dataset.

    Parameters
    ----------
    datapath : str
        Path to dataset directory

    da: boolean
        Enables or disable data augmentation

    Returns
    -------
    model
        builded model
    """
    # # system config: seed
    # keras.backend.clear_session()
    # tf.random.set_seed(42)
    # np.random.seed(42)

    model = unet(5)

    return model


def create_callbacks(run_logdir, checkpoint_path="model.h5", patience=2, early_stop=False):
    """Creates a tab composed of defined callbacks.

    Early stopping is disabled by default.

    All checkpoints saved by tensorboard will be stored in a new directory
    named /logs in main folder.
    The final .h5 file will also be stored in a new directory named /models.

    Parameters
    ----------
    run_logdir : str
        Path to logs directory, create a new one if it doesn't exist.

    checkpoint_path : str
        Path to model directory, create a new one if it doesn't exist.

    early_stop : boolean (False by default)
        Enables or disables early stopping.

    Returns
    -------
    list
        a list of defined callbacks
    """

    callbacks = []

    if early_stop:
        print(f"Early stopping patience : {patience}")
        early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_loss",
                                                          patience=patience,
                                                          #mode="auto",
                                                          restore_best_weights=True,
                                                          verbose=1)
        callbacks.append(early_stopping_cb)

    checkpoint_cb = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                    save_best_only=True)
    callbacks.append(checkpoint_cb)

    tensorboard_cb = keras.callbacks.TensorBoard(log_dir=run_logdir,
                                                 histogram_freq=1,
                                                 write_images=True)
    callbacks.append(tensorboard_cb)

    now = datetime.now()
    csvlogger_cb = keras.callbacks.CSVLogger(filename=f"./csv_logs/training_{now.strftime('%m_%d_%H_%M')}.csv",
                                             separator=",",
                                             append=True)
    callbacks.append(csvlogger_cb)

    backup_restore_cb = keras.callbacks.BackupAndRestore(backup_dir="./tmp/backup")
    callbacks.append(backup_restore_cb)

    return callbacks


def main():
    """Main function."""
    now = datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", "-e", type=int, default=10,
                        help="custom epochs number")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="custom learning rate")
    parser.add_argument("--load", default=False,
                        help="load previous model",
                        action="store_true")
    parser.add_argument("--modelpath", default="/models/run1.h5",
                        help="path to .h5 file for transfert learning")
    parser.add_argument("--batch", "-b", type=int, default=16,
                        help="batch size")
    parser.add_argument("--datapath", help="path to the dataset")
    parser.add_argument("--log", default=f"logs/run{now.strftime('%m_%d_%H_%M')}", help="set path to logs")
    parser.add_argument("--checkpoint", "-c", default=f"models/run{now.strftime('%m_%d_%H_%M')}.h5",
                        help="set checkpoints path and name")

    args = parser.parse_args()

    datapath = args.datapath
    load_model = args.load
    epochs = args.epochs
    lr = args.lr
    logpath = args.log
    cppath = args.checkpoint
    bs = args.batch

    # data loading
    path = os.path.join(datapath)

    train_set, val_set, test_set, NUM_TRAIN, NUM_TEST = create_pipeline_performance(path)

    print(f"{NUM_TRAIN} images for train")
    print(f"{NUM_TEST} images for test")

    # model building
    model_name = "unet"
    if load_model:
        model_name = "custom"
        model_path = args.modelpath
        model = keras.models.load_model(model_path, compile=False)
        print(f"Transfert learning from {model_path}")
    else:
        model = model_builder()

    decay_steps = (NUM_TRAIN // bs) * epochs
    lr_scheduler = keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=lr,
                                                              decay_steps=decay_steps,
                                                              end_learning_rate=0.000001,
                                                              power=1.5)

    losses = ["mae"]
    metrics = ["accuracy"]

    optimizer = keras.optimizers.Adam(learning_rate=lr_scheduler)
    model.compile(loss=losses,
                  optimizer=optimizer,
                  metrics=metrics)

    model.summary()

    model_architecture_path = "architecture/"
    if os.path.exists(model_architecture_path) is False:
        os.makedirs(model_architecture_path)

    keras.utils.plot_model(model,
                           to_file=os.path.join(model_architecture_path,
                                                f"model_unet_{model_name}_{now.strftime('%m_%d_%H_%M')}.png"),
                           show_shapes=True)

    # callbacks
    run_logs = logpath
    checkpoint_path = cppath
    if epochs < 40:
        cb_patience = 5
    else:
        cb_patience = epochs // 10
    # cb = create_callbacks(run_logs, checkpoint_path, cb_patience, True)
    cb = create_callbacks(run_logs, checkpoint_path, 150, True)

    EPOCH_STEP_TRAIN = NUM_TRAIN // bs
    EPOCH_STEP_TEST = NUM_TEST // bs

    # training and evaluation
    model.fit(x=train_set,
              epochs=epochs,
              verbose=1,
              callbacks=cb,
              validation_data=val_set,
              steps_per_epoch=EPOCH_STEP_TRAIN,
              validation_steps=EPOCH_STEP_TEST)


    _, mse_metrics = model.evaluate(x=test_set,
                                    steps=EPOCH_STEP_TEST)
    print("Mean Squared Error : {:.03f}".format(mse_metrics))


if __name__ == "__main__":
    main()