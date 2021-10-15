import logging
import json
from time import time

from flaml import AutoML, __version__, ml

from frameworks.shared.callee import call_run, result, output_subdir
from frameworks.shared.utils import Timer

import numpy as np
from sklearn.metrics import roc_auc_score, r2_score

log = logging.getLogger(__name__)

from numpy import random


def run(dataset, config):
    random.seed(6713392)
    log.info(f"\n**** Dummy [v0.0.1] ****\n")

    t1 = time()

    X_train, y_train = dataset.train.X, dataset.train.y.squeeze()
    X_test, y_test = dataset.test.X, dataset.test.y.squeeze()

    is_classification = config.type == "classification"

    if "extract-meta" in config.framework_params:
        mf = extract_metafeatures(X_train, y_train, is_classification)
        metaf_file = config.framework_params["extract-meta"]
        with open(metaf_file, "a+") as f:
            f.write(config.name + "," + ",".join(str(x) for x in mf) + "\n")

    if y_train.dtype == "category":
        y_train = y_train.cat.codes
        y_test = y_test.cat.codes

    model_json = config.framework_params["model"]
    with open(model_json, "r") as f:
        stored = json.load(f)

    stored["hyperparameters"].pop("objective", None)
    stored["hyperparameters"].pop("task", None)

    mapping = {"binary": "binary", "multiclass": "multi", "regression": "regression"}

    class_ = ml.get_estimator_class(config.type, stored["flaml-name"])
    M = class_(**stored["hyperparameters"], task=mapping[config.type_])

    automl = AutoML()

    automl.fit(X_train, y_train, max_iter=0, keep_search_state=True, task=config.type)
    # read from automl task type, set params.
    M.__class__.init()

    t3 = time()
    M.fit(
        automl._X_train_all,
        automl._y_train_all,
        budget=config.max_runtime_seconds - (t3 - t1),
    )
    automl._trained_estimator = M
    t4 = time()

    print(M._n_estimators)

    predictions = automl.predict(X_test)
    if is_classification:
        probabilities = automl.predict_proba(X_test)
        if config["type_"] == "binary":
            score = 1 - roc_auc_score(y_test, probabilities[:, 1], multi_class="ovo")
        else:
            score = 1 - roc_auc_score(y_test, probabilities, multi_class="ovo")
    else:
        probabilities = None
        score = 1 - r2_score(y_test, predictions)

    t5 = time()

    r = open(config.framework_params["output"], "a+")
    r.write(f"{model_json},{config.name},{config.fold},{score}\n")
    r.close()

    labels = automl.classes_ if is_classification else None

    return result(
        output_file=config.output_predictions_file,
        probabilities=probabilities,
        predictions=predictions,
        truth=y_test,
        models_count=1,
        training_duration=t4 - t3,
        predict_duration=t5 - t4,
        probabilities_labels=labels,
        model_name=model_json,
    )


def extract_metafeatures(X, y, classification):
    n_row = X.shape[0]
    n_feat = X.shape[1]
    n_class = y.nunique() if classification else 0
    pct_num = X.select_dtypes(include=np.number).shape[1] / n_feat
    return (n_row, n_feat, n_class, pct_num)


if __name__ == "__main__":
    call_run(run)
