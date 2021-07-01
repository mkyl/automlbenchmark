import logging
import os
import pickle
from pickle import HIGHEST_PROTOCOL

from flaml import AutoML, __version__

from frameworks.shared.callee import call_run, result, output_subdir
from frameworks.shared.utils import Timer

from sklearn.metrics import roc_auc_score

log = logging.getLogger(__name__)


def run(dataset, config):
    log.info(f"\n**** Dummy [v0.0.1] ****\n")

    X_train, y_train = dataset.train.X, dataset.train.y.squeeze()
    X_test, y_test = dataset.test.X, dataset.test.y.squeeze()

    is_classification = config.type == 'classification'
    
    if dataset.problem_type != "binary":
        return {}

    r = open("auc-matrix.csv", "a")

    for file in os.listdir("mined/"):
        with open("mined/" + file, 'rb') as f:
            M = pickle.load(f, encoding="utf-8")
            M.fit(X_train, y_train)
            P = M.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, P)
            r.write(f"{file}, {config.name}, {auc}\n")

    r.close()
   
    return {}


if __name__ == '__main__':
    call_run(run)
