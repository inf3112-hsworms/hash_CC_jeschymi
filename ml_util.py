##########################################################
#                      Machine Learning tools
##########################################################

import os
import random
import warnings
from copy import copy, deepcopy
from datetime import datetime
from os import listdir
from typing import Iterable

import numpy
from numpy import float64, mean
from scipy import signal
from scipy.stats import moment, skew, kurtosis
from sklearn.model_selection import train_test_split
from sparse import COO
import numpy as np
import pandas
from joblib import Parallel, delayed

from CC_noise import add_cc_noise
from tail_estimation.tail_estimation import moments_estimator, hill_estimator, smooth_hill_estimator, \
    kernel_type_estimator, pickands_estimator


def modify_data(df, hashfunc):
    if not isinstance(hashfunc, Iterable):
        df["modified"], df["channel_noise"] = add_cc_noise(df["ping time"], hashfunc)
        for x in range(20):
            df["modified_" + str(x)], df["channel_noise_" + str(x)] = add_cc_noise(df["ping time"], hashfunc)
    else:
        for hashf in hashfunc:
            for x in range(4):
                df[hashf.__name__ + "_" + str(x)], df[hashf.__name__ + "_cc_" + str(x)] = add_cc_noise(df["ping time"], hashf)


def read_and_modify(file, hashfunc, scale_factor=1, schmidbauer=False, savemode=False, save_to=""):
    df = pandas.read_csv(file)
    df = df.rename(columns=lambda x: x.strip())
    print(file)
    if len(df)<10:
        return None
    if schmidbauer:
        df["ping time"] = df["Time since previous frame in this TCP stream"]
        df["avg traceroute"] = 0
    df["ping time"] = df["ping time"] * scale_factor
    modify_data(df, hashfunc)
    if not savemode:
        return df
    else:
        df.to_csv(save_to + "/" + os.path.basename(file))
        print("successfully saved")
        return df

def read_already_modified_from(files):
    return [pandas.read_csv(file) for file in files]


def get_and_modify_data_from(files, hashfunc, scale_factor=1, savemode=False, save_to=""):
    data_frames = Parallel(n_jobs=7)(delayed(
        read_and_modify)(file, hashfunc, scale_factor=scale_factor, savemode=savemode, save_to=save_to) for file in files)
    return data_frames



def get_and_modify_data_from_schmidbauer(hashfunc):
    data_frames = Parallel(n_jobs=7)(delayed(
        read_and_modify)(file, hashfunc, schmidbauer=True) for file in
        ["data/00_0/" + f for f in listdir("data/00_0") if f.endswith(".csv")])
    return data_frames


def get_datasets(df, windows, shuffle = True, max_per_window=15):
    if df is None:
        return []
    if shuffle:
        data_c = df.sample(frac=1)
    else:
        data_c = df
    sets = []
    for window in windows:
        maximum = max_per_window * window
        i = 0
        while i < len(data_c) and i < maximum:
            sets.append(data_c[i:i+window])
            i += window
        data_c = df.sample(frac=1)
    return sets


def create_histogram(data, min=0, max=1, bins=2500):
    hist_, scale = np.histogram(data, bins=bins, range=[min, max])
    return COO.from_numpy(
            np.array([np.array([h/len(data)]) for h in hist_])
        )


def parallelizable_histogram_creation(df, prefix = "modified_", choose=False):
    choice = random.choice(range(3))
    hist_X_0 = create_histogram(df["ping time"])
    hist_X_1 = create_histogram(df[prefix + str(choice) if not choose else prefix])
    return hist_X_0, hist_X_1


def parallelizable_cutoff(df, prefix = "modified_", window=250):
    if len(df)==window:
        choice = random.choice(range(3))
        X_0 = df["ping time"]
        X_1 = df[prefix + str(choice)]
        return X_0, X_1
    else:
        return None, None

def statistical_data(series, slice, prefix="", bootstrap=False, hist_range=[0, 1]):

    sorted = np.sort(slice)[::-1]
    l = lambda arr: arr[1][int(len(arr[1])/2)] if len(arr[1])>1 else 0

    hill = hill_estimator(sorted, r_bootstrap=50, bootstrap=bootstrap)
    series[prefix + "hill_est"] = l(hill)

    pickands = pickands_estimator(sorted)
    series[prefix + "pickands_est"] = l(pickands)

    series[prefix + "mean"] = mean(sorted)

    hist = create_histogram(sorted, min=hist_range[0], max=hist_range[1]).todense()
    scale_const = (hist_range[1] - hist_range[0]) / 10000
    modal = np.argmax(hist) * scale_const
    series[prefix + "mode"] = modal
    series[prefix + "std"] = np.std(sorted)
    series[prefix + "log_std"] = np.std(np.log(np.abs(sorted)))
    series[prefix + "median"] = np.median(sorted)
    series[prefix + "skew"] = skew(sorted)
    series[prefix + "skew_inverse"] = 1 / series[prefix + "skew"]
    series[prefix + "kurtosis"] = kurtosis(sorted)
    series[prefix + "kurtosis_inverse"] = 1 / series[prefix + "kurtosis"]
    series[prefix + "FWHM"]= scale_const * signal.peak_widths([h[0] for h in hist], [np.argmax(hist)])[0][0]
    series[prefix + "hist_squared_sum"] = np.sum([h[0] ** 2 for h in hist])
    series[prefix + "suessmann"] = 1/ series[prefix + "hist_squared_sum"]
    series[prefix + "max_hist"] = np.max(hist)
    series[prefix + "max_hist_inverse"] = 1/series[prefix + "max_hist"]
    series[np.isnan(series)] = 0


def get_suessmann(data):
    hist_range = [0, 1.5]
    hist = create_histogram(data, min=hist_range[0], max=hist_range[1]).todense()
    return 1/np.sum([h[0] ** 2 for h in hist])


def create_statistical_data(slice):
    warnings.filterwarnings("ignore")

    series = pandas.Series({}, dtype=float64)

    statistical_data(series, slice)

    return series


def parallelizable_stats_creation(df, prefix="modified_", choose=False):
    choice = random.choice(range(3))

    if len(df) < 10:
        return None, None

    stat_X_0 = create_statistical_data(df["ping time"])
    stat_X_1 = create_statistical_data(df[prefix + str(choice) if not choose else prefix])
    return stat_X_0, stat_X_1


def get_X_0_X_1(df, windows, editfunc, shuffle=True):
        datasets = get_datasets(df, windows, shuffle=shuffle)
        X_0_X_1 = Parallel(n_jobs=7)(delayed(editfunc)
                                   (df) for df in datasets)
        return X_0_X_1


def ml_train(data, model, editfunc, savename_prefix,
             windows=list(range(20, 70, 10)) + list(range(50, 300, 50)),
             keras=True, sparse=False,
             **kwargs):
    X_0, X_1 = [], []
    for df in data:
        X_0_X_1 = get_X_0_X_1(df, windows, editfunc)
        for hists_X_0, hists_X_1 in X_0_X_1:
            if hists_X_0 is not None and hists_X_1 is not None:
                X_0.append(hists_X_0)
                X_1.append(hists_X_1)
        del df
    del data
    print("ready for ML")
    y_0 = len(X_0)
    y_1 = len(X_1)
    print(f"lens: {y_0}, {y_1}")
    X = X_0 + X_1
    y = np.concatenate((np.array([0] * y_0), np.array([1] * y_1)))
    #y = y.reshape(1, -1)
    print("-----------------")
    if sparse:
        X = [x.todense() for x in X]
        if not keras:
            X = [[x__[0] for x__ in x_] for x_ in X]
    print("done array casting")
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    print(len(X))
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    if keras:
        model.fit(X_train, y_train, validation_data=(X_test, y_test), **kwargs)
        model.save(savename_prefix + datetime.now().strftime("%Y_%m_%d_%H_%M"))
    else:
        model.fit(X_train, y_train)
        accuracy = model.score(X_train, y_train)
        val_accuracy = model.score(X_test, y_test)
        print("accuracy: {0}; val_accuracy: {1}".format(accuracy, val_accuracy))
    return model