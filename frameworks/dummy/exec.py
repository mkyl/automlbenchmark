import logging
import os
import pickle
from pickle import HIGHEST_PROTOCOL

from flaml import AutoML, __version__

from frameworks.shared.callee import call_run, result, output_subdir
from frameworks.shared.utils import Timer

from sklearn.metrics import roc_auc_score
from lightgbm.basic import LightGBMError

import numpy as np

log = logging.getLogger(__name__)


def run(dataset, config):
    log.info(f"\n**** Dummy [v0.0.1] ****\n")

    X_train, y_train = dataset.train.X, dataset.train.y.squeeze()
    X_test, y_test = dataset.test.X, dataset.test.y.squeeze()

    if y_train.dtype == "category":
        y_train = y_train.cat.codes
        y_test = y_test.cat.codes

    for file in os.listdir("mined/"):
        with open("mined/" + file, 'rb') as f:
            try:
                M = pickle.load(f, encoding="utf-8")
                try:
                    M._n_classes = len(np.unique(dataset.train.y.squeeze()))
                except:
                    print(type(M))
                    pass
                M.fit(X_train, y_train)
                P = M.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, P)
                r = open(f"auc-matrix.csv", "a+")
                r.write(f"{file},{config.name},{auc}\n")
                r.close()
            except Exception as e:
                continue

    return {}


if __name__ == '__main__':
    call_run(run)
