import os
import time
import json
import numpy as np
import functools

import argparse

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler

import keras.backend as K


def get_data(path, dataset="protein_structure"):
    train = np.array(np.load(os.path.join(path, "%s_train_data.npy" % dataset)), dtype=np.float64)
    train_targets = np.array(np.load(os.path.join(path, "%s_train_targets.npy" % dataset)), dtype=np.float64)

    valid = np.array(np.load(os.path.join(path, "%s_valid_data.npy" % dataset)), dtype=np.float64)
    valid_targets = np.array(np.load(os.path.join(path, "%s_valid_targets.npy" % dataset)), dtype=np.float64)

    test = np.array(np.load(os.path.join(path, "%s_test_data.npy" % dataset)), dtype=np.float64)
    test_targets = np.array(np.load(os.path.join(path, "%s_test_targets.npy" % dataset)), dtype=np.float64)

    m = np.mean(train, axis=0)
    s = np.std(train, axis=0)

    train = (train - m) / s
    valid = (valid - m) / s
    test = (test - m) / s

    m = np.mean(train_targets, axis=0)
    s = np.std(train_targets, axis=0)

    train_targets = (train_targets - m) / s
    valid_targets = (valid_targets - m) / s
    test_targets = (test_targets - m) / s
    return train, train_targets, valid, valid_targets, test, test_targets


def mean_squared_err(y_true, y_pred):
    return K.mean((y_true - y_pred)**2)


def fix(epoch, initial_lr):
    return initial_lr


def exponential(epoch, initial_lr, T_max, decay_rate=0.96):
    return initial_lr * decay_rate ** (epoch / T_max)


def cosine(epoch, initial_lr, T_max):
    final_lr = 0
    return final_lr + (initial_lr - final_lr) / 2 * (1 + np.cos(np.pi * epoch / T_max))


def main(args):
    """Builds, trains, and evaluates the model."""

    x_train, y_train, x_valid, y_valid, x_test, y_test = get_data(path=args["data_dir"],
                                                                  dataset=args["dataset"])

    model = Sequential()
    model.add(Dense(args["n_units_1"], activation=args["activation_fn_1"], input_dim=x_train.shape[1]))
    model.add(Dropout(args["dropout_1"]))
    model.add(Dense(args["n_units_2"], activation=args["activation_fn_2"]))
    model.add(Dropout(args["dropout_2"]))
    model.add(Dense(1, activation='linear'))

    adam = Adam(lr=args["init_lr"])
    model.compile(loss='mean_squared_error',
                  optimizer=adam,
                  metrics=[mean_squared_err])

    if args["lr_schedule"] == "cosine":
        schedule = functools.partial(cosine, initial_lr=args["init_lr"], T_max=args["n_epochs"])

    if args["lr_schedule"] == "exponential":
        schedule = functools.partial(exponential, initial_lr=args["init_lr"], T_max=args["n_epochs"])

    elif args["lr_schedule"] == "const":
        schedule = functools.partial(fix, initial_lr=args["init_lr"])

    lrate = LearningRateScheduler(schedule)

    callbacks_list = [lrate]
    st = time.time()

    hist = model.fit(x_train, y_train,
                     epochs=args["n_epochs"],
                     batch_size=args["batch_size"],
                     validation_data=(x_valid, y_valid),
                     callbacks=callbacks_list,
                     verbose=2)

    final_perf = model.evaluate(x_test, y_test, batch_size=args["batch_size"])[1]

    config = {
            "n_units_1": args["n_units_1"],
            "n_units_2": args["n_units_2"],
            "dropout_1": args["dropout_1"],
            "dropout_2": args["dropout_2"],
            "activation_fn_1": args["activation_fn_1"],
            "activation_fn_2": args["activation_fn_2"],
            "init_lr": args["init_lr"],
            "lr_schedule": args["lr_schedule"],
            "batch_size": args["batch_size"]}

    r = dict()
    r["config"] = config
    r["runtime"] = time.time() - st
    r["train_loss"] = hist.history["loss"]
    r["valid_loss"] = hist.history["val_loss"]
    r["train_mse"] = hist.history["mean_squared_err"]
    r["valid_mse"] = hist.history["val_mean_squared_err"]
    r["final_test_error"] = final_perf
    r["n_params"] = int(model.count_params())

    os.makedirs(args["model_dir"], exist_ok=True)
    json.dump(r, open(os.path.join(args["model_dir"], "result.json"), "w"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=128, type=int, nargs='?', help='batch size')
    parser.add_argument('--n_units_1', default=16, type=int, nargs='?', help='number of units in first layer')
    parser.add_argument('--n_units_2', default=16, type=int, nargs='?', help='number of units in second layer')
    parser.add_argument('--dropout_1', default=0, type=float, nargs='?', help='dropout in first layer')
    parser.add_argument('--dropout_2', default=0, type=float, nargs='?', help='dropout in second layer')
    parser.add_argument('--activation_fn_1', default="tanh", nargs='?', type=str, help='type of activation function '
                                                                                       'in first layer')
    parser.add_argument('--activation_fn_2', default="tanh", nargs='?', type=str, help='type of activation function '
                                                                                       'in second layer')
    parser.add_argument('--init_lr', default=1e-3, type=float, nargs='?', help='initial learning rate')
    parser.add_argument('--lr_schedule', default="cosine", nargs='?', type=str, help='learning rate schedule')
    parser.add_argument('--n_epochs', default=100, type=int, nargs='?', help='total number of epochs for training')

    parser.add_argument('--model_dir', nargs='?', default="./model_dir", type=str, help='specifies the directory where'
                                                                                        'the results will be saved')
    parser.add_argument('--data_dir', nargs='?', default="./protein_structure", type=str,
                        help='path to the dataset')
    parser.add_argument('--dataset', default="protein_structure", nargs='?', type=str, help='name of the dataset')

    args = vars(parser.parse_args())

    main(args)
