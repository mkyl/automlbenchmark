import logging
import os
import pickle
from pickle import HIGHEST_PROTOCOL
from time import time

from flaml import AutoML, __version__

from frameworks.shared.callee import call_run, result, output_subdir
from frameworks.shared.utils import Timer

from sklearn.metrics import roc_auc_score, log_loss, balanced_accuracy_score
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
                automl = AutoML()
                M = pickle.load(f, encoding="utf-8")
                if hasattr(M, "params") and "objective" in M.params:
                    if config['type_'] == "binary":
                        M.params["objective"] = "binary"
                    else:
                        M.params["objective"] = "multiclass"
                automl.fit(X_train, y_train, max_iter=0)
                # read from automl task type, set params.
                M.__class__.init()
                t1 = time()
                print(len(automl._X_train_all))
                M.fit(automl._X_train_all, automl._y_train_all, budget=600)
                t2 = time()
                automl._trained_estimator = M
                if config['type_'] == "binary":
                    P = automl.predict_proba(X_test)[:, 1]
                    score = 1 - roc_auc_score(y_test, P)
                else:
                    P = automl.predict_proba(X_test)
                    score = log_loss(y_test, P)
                r = open(f"regret-matrix.csv", "a+")
                r.write(f"{file},{config.name},{score}\n")
                r.close()

                t = open(f"timing-matrix.csv", "a+")
                t.write(f"{file},{config.name},{t2 - t1}\n")
                t.close()
            except Exception as e:
                import traceback
                print(M.__class__)
                print(type(M))
                print(file)
                print(traceback.format_exc())
                continue

    import sys
    sys.exit(0)


if __name__ == '__main__':
    call_run(run)
