import logging
import json
from time import time

from flaml import AutoML, __version__, model

from frameworks.shared.callee import call_run, result, output_subdir
from frameworks.shared.utils import Timer


from sklearn.metrics import roc_auc_score
from lightgbm.basic import LightGBMError

log = logging.getLogger(__name__)

from numpy import random


def run(dataset, config):
    random.seed(6713392)
    log.info(f"\n**** Dummy [v0.0.1] ****\n")

    X_train, y_train = dataset.train.X, dataset.train.y.squeeze()
    X_test, y_test = dataset.test.X, dataset.test.y.squeeze()

    if y_train.dtype == "category":
        y_train = y_train.cat.codes
        y_test = y_test.cat.codes

    model_json = config.framework_params["model"]
    with open(model_json, "r") as f:
        stored = json.load(f)

    class_ = getattr(model, stored["class"])
    print(class_)
    M = class_(**stored["hyperparameters"])

    automl = AutoML()
    if hasattr(M, "params") and "objective" in M.params:
        if config["type_"] == "binary":
            M.params["objective"] = "binary"
        else:
            M.params["objective"] = "multiclass"
    automl.fit(X_train, y_train, max_iter=0, keep_search_state=True)
    # read from automl task type, set params.
    M.__class__.init()
    t1 = time()
    M.fit(automl._X_train_all, automl._y_train_all, budget=config.max_runtime_seconds)
    t2 = time()
    automl._trained_estimator = M

    P = automl.predict_proba(X_test)
    if config["type_"] == "binary":
        P = P[:, 1]
    score = 1 - roc_auc_score(y_test, P, multi_class="ovo")

    r = open(config.framework_params["output"], "a+")
    r.write(f"{model_json},{config.name},{score}\n")
    r.close()

    return result()


if __name__ == "__main__":
    call_run(run)
