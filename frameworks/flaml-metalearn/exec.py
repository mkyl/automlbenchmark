import logging
import os
import json

from flaml import AutoML, __version__

from frameworks.shared.callee import call_run, result, output_subdir
from frameworks.shared.utils import Timer

from numpy import random
import pandas as pd
import numpy as np

log = logging.getLogger(__name__)


def run(dataset, config):
    random.seed(539492)

    log.info(f"\n**** FLAML [v{__version__}] ****\n")

    X_train, y_train = dataset.train.X, dataset.train.y.squeeze()
    X_test, y_test = dataset.test.X, dataset.test.y.squeeze()

    is_classification = config.type == "classification"
    time_budget = config.max_runtime_seconds
    n_jobs = config.framework_params.get("_n_jobs", config.cores)
    log.info("Running FLAML with {} number of cores".format(config.cores))
    aml = AutoML()

    # Mapping of benchmark metrics to flaml metrics
    metrics_mapping = dict(
        acc="accuracy",
        auc="roc_auc",
        f1="f1",
        logloss="log_loss",
        mae="mae",
        mse="mse",
        rmse="rmse",
        r2="r2",
    )
    perf_metric = (
        metrics_mapping[config.metric] if config.metric in metrics_mapping else "auto"
    )
    if perf_metric is None:
        log.warning("Performance metric %s not supported.", config.metric)

    training_params = {
        k: v for k, v in config.framework_params.items() if not k.startswith("_")
    }

    if "metafeatures" in training_params:
        meta_f = extract_metafeatures(X_train, y_train)
        # M = pd.read_csv(training_params["metafeatures"], index_col=[0], header=[0])
        # M = M[F]
        # meta_f = M.loc[config.name]
    else:
        meta_f = None

    if "portfolio" in training_params:
        with open(training_params["portfolio"], "r") as f:
            portfolio = json.load(f)
    else:
        # TODO load default portfolio
        pass

    metalearn = training_params["metalearn"]
    if metalearn.lower() == "none":
        metalearn = None

    max_iter = (
        int(training_params["max_iter"]) if "max_iter" in training_params else 1000000
    )

    log_dir = output_subdir("logs", config)
    flaml_log_file_name = os.path.join(log_dir, "flaml.log")
    with Timer() as training:
        aml.fit(
            X_train,
            y_train,
            metric=perf_metric,
            task=config.type,
            n_jobs=n_jobs,
            log_file_name=flaml_log_file_name,
            time_budget=time_budget,
            max_iter=max_iter,
            retrain_full=False,
            portfolio=portfolio,
            metafeatures=meta_f,
            metalearn=metalearn,
            # **training_params
        )

    with Timer() as predict:
        predictions = aml.predict(X_test)
    probabilities = aml.predict_proba(X_test) if is_classification else None
    labels = aml.classes_ if is_classification else None
    return result(
        output_file=config.output_predictions_file,
        probabilities=probabilities,
        predictions=predictions,
        truth=y_test,
        models_count=len(aml.config_history),
        training_duration=training.duration,
        predict_duration=predict.duration,
        probabilities_labels=labels,
    )


def extract_metafeatures(X, y):
    n_row = X.shape[0]
    n_feat = X.shape[1]
    # TODO set to 0 if regression task?
    n_class = y.nunique()
    pct_num = X.select_dtypes(include=np.number).shape[1] / n_feat
    return (n_row, n_feat, n_class, pct_num)


if __name__ == "__main__":
    call_run(run)
