
import numpy as np
from keras import Sequential
from keras.layers import Conv1D, Flatten, Dense
from keras.losses import BinaryCrossentropy
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from ml_util import parallelizable_stats_creation, read_and_modify, get_X_0_X_1, parallelizable_histogram_creation


def get_model_ml_histogram_direct(bins=2500):
    model = Sequential()
    # convolutional layer
    model.add(
        Conv1D(50, 2, padding='valid', activation='relu', input_shape=(bins, 1))
    )
    #model.add(MaxPool1D(pool_size=10))
    model.add(Flatten())
    # hidden layer
    model.add(Dense(50, activation='sigmoid'))
    model.add(Dense(25, activation='sigmoid'))
    # output layer
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss=BinaryCrossentropy(),
                  metrics=['accuracy'], optimizer='adam')

    return model


def ml_statistical_measures():
    return DecisionTreeClassifier(max_depth=10, class_weight={0:1.5, 1:1})


def get_gradient_boost():
    return GradientBoostingClassifier(max_depth=5, n_iter_no_change=3, n_estimators=20)

def get_random_forest():
    return RandomForestClassifier(n_jobs=7, n_estimators=20, max_depth=5, class_weight={0:1, 1:1.5})


def test_with_data(model, file, hashfunc, name="", windows=[100], keras=False):
    testdata = read_and_modify(file, hashfunc)
    stat_data = get_X_0_X_1(testdata, windows, parallelizable_stats_creation if not keras else parallelizable_histogram_creation)
    X_0 = [a[0] for a in stat_data]
    X_1 = [a[1] for a in stat_data]
    none_filter = lambda x: x is not None
    X_0 = list(filter(none_filter, X_0))
    X_1 = list(filter(none_filter, X_1))
    _, X, _, y = train_test_split(X_0 + X_1, [0]*len(X_0) + [1]*len(X_1), train_size=1)
    if not keras:
        y_pred = model.predict(X)
        print("accuracy {1}: {0}".format(model.score(X, y), name))
        print(confusion_matrix(y, y_pred, normalize="all"))
    else:
        X = np.array([x.todense() for x in X])
        score, acc = model.evaluate(X, np.array(y))
        y_pred = model.predict(X)
        y_pred = [np.round(y_) for y_ in y_pred]
        print(confusion_matrix(y, y_pred, normalize="all"))
        print("accuracy {0}: {1} (score: {2})".format(name, acc, score))