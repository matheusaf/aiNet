import sys
import csv
import logging as lg
import os
from collections.abc import Callable
from datetime import datetime
from functools import partial
from time import time
from typing import Any
from traceback import format_exc

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import davies_bouldin_score
from torch import cuda

import representations as representation_models
import utils.evaluations.clustering_metrics as cmetrics
from models import AiNet
from utils import cosine_distances, euclidean_distances, print_progress_bar


class AiNetExecutor:
    __slots__ = ["__logger_"]

    __normed_representations_ = {
        "NGram": True,
        "LIWC": True,
        "MRC2": True,
        "STagger": True,
        "Word2Vec": True,
        "Doc2Vec": True,
        "FastText": True,
        "SBert": True,
    }

    def __init__(self, logger: lg.Logger) -> None:
        init_start = time()
        self.__logger_ = logger

        assert logger is not None, "logger cannot be None"

        self.__logger_.debug(
            "Instantiating {%s} took %0.5f second(s)", self.__class__.__name__, time() - init_start
        )

    def __instantiate_ainet_(self, execution_plan: dict) -> AiNet:
        instance_start = time()
        model_params: dict = execution_plan.get("model_params", {})

        assert model_params is not None and len(model_params), "model_params cannot be None"

        self.__logger_.debug(
            "creating a new aiNet instance with the following params %s took %0.5f second(s)",  # noqa: E501
            str(model_params),
            time() - instance_start,
        )

        return AiNet(logger=self.__logger_, **model_params)

    def __read_file_(
        self, dataset_path: str, text_column: str, label_column: str, delimiter: str
    ) -> pd.DataFrame:
        assert dataset_path is not None and len(dataset_path), "dataset_path cannot be None"
        assert text_column is not None and len(text_column), "text_column cannot be None"
        assert label_column is not None and len(label_column), "label_column cannot be None"
        assert delimiter is not None and len(delimiter), "delimiter cannot be None"

        abs_dataset_path = os.path.abspath(dataset_path)

        assert os.path.exists(abs_dataset_path), f"file '{abs_dataset_path}' does not exist"

        read_file_start = time()

        csv.field_size_limit(sys.maxsize)

        with open(abs_dataset_path, "r+", encoding="utf-8", newline="") as csv_file:
            self.__logger_.debug(
                "opening file '%s' took %0.5f second(s)", abs_dataset_path, time() - read_file_start
            )

            read_file_start = time()

            reader = csv.reader(
                csv_file, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL, quotechar='"'
            )

            self.__logger_.debug(
                "reading file '%s' with csv reader took %0.5f second(s)",
                abs_dataset_path,
                time() - read_file_start,
            )

            read_file_start = time()

            dataset_data = [row for row in reader]

            assert dataset_data is not None or len(dataset_data) > 0, "Failed to read file"

            self.__logger_.info(
                "finished reading file '%s' with %d line(s) using csv reader took %0.5f second(s)",
                abs_dataset_path,
                len(dataset_data),
                time() - read_file_start,
            )

            dataset_header = dataset_data.pop(0)

            dataset = pd.DataFrame(columns=dataset_header, data=dataset_data)

            dataset = dataset[[text_column, *label_column]]

            return dataset

    def __generate_representation_(
        self,
        representation_name: str,
        representation_model_dict: dict,
        dataset_dataframe: pd.DataFrame,
        text_column: str,
    ) -> tuple[Any, Any, dict]:
        assert (
            representation_model_dict is not None
            and len(representation_model_dict)
            and len(representation_model_dict[representation_name].values())
        ), "representation_params cannot be None"

        representation_model_dict = representation_model_dict.get(representation_name, {}).get(
            "representation_params", {}
        )

        representation_start = time()

        cuda.empty_cache()

        self.__logger_.debug(
            "cleaning cuda cache took %0.5f second(s)", time() - representation_start
        )

        final_representation: representation_models.Representation | None = None

        if trained_model := representation_model_dict.get("trained_model", False):
            representation_start = time()

            if representation_name == "Word2Vec":
                final_representation = representation_models.Word2Vec.from_trained_model(
                    trained_model
                )
            if representation_name == "FastText":
                final_representation = representation_models.FastText.from_facebook_model(
                    trained_model
                )

            self.__logger_.debug(
                "reading trained_model '%s' for representation model %s took %0.5f second(s)",  # noqa: E501
                trained_model,
                representation_name,
                time() - representation_start,
            )

        else:
            if representation_name in {
                "Word2Vec",
                "Doc2Vec",
                "FastText",
            } and not representation_model_dict.get("train_corpus", False):
                representation_model_dict["train_corpus"] = dataset_dataframe.iloc[:, 0].tolist()

            self.__logger_.debug(
                "generating representation model %s with parameters %s",  # noqa: E501
                representation_name,
                str(
                    {
                        **representation_model_dict,
                        "train_corpus": True
                        if representation_model_dict.get("train_corpus", False)
                        else False,
                    }
                ),
            )

            if representation_name == "NGram":
                final_representation = representation_models.NGram(**representation_model_dict)
            if representation_name == "LIWC":
                final_representation = representation_models.LIWC(**representation_model_dict)
            if representation_name == "STagger":
                final_representation = representation_models.STagger(**representation_model_dict)
            if representation_name == "MRC2":
                final_representation = representation_models.MRC2(**representation_model_dict)
            if representation_name == "Word2Vec":
                final_representation = representation_models.Word2Vec(**representation_model_dict)
            if representation_name == "Doc2Vec":
                final_representation = representation_models.Doc2Vec(**representation_model_dict)
            if representation_name == "FastText":
                final_representation = representation_models.FastText(**representation_model_dict)
            if representation_name == "SBert":
                final_representation = representation_models.SBert(**representation_model_dict)

        assert final_representation is not None, f"{representation_name} is not valid"

        self.__logger_.debug(
            "instantiating representation model '%s' with shape (%d, %d) took %0.5fs",  # noqa: E501
            representation_name,
            dataset_dataframe.shape[0],
            dataset_dataframe.shape[1],
            time() - representation_start,
        )

        features, representation = final_representation.generate_representation(
            dataset_dataframe[text_column].tolist()
        )

        self.__logger_.debug(
            "finishing generating representation model '%s' with shape (%d, %d) took %0.5f second(s)",  # noqa: E501
            representation_name,
            representation.shape[0],  # type: ignore
            representation.shape[1],  # type: ignore
            time() - representation_start,
        )

        return (features, representation, representation_model_dict)

    def __calculate_external_metrics_(
        self,
        edist: np.ndarray,
        cdist: np.ndarray,
        label_column: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict:
        external_results = dict()
        external_results["label_column"] = label_column

        external_metrics_to_calculate = {
            "nmi_geometric": partial(
                cmetrics.nmi, **{"y_true": y_true, "y_pred": y_pred, "average_method": "geometric"}
            ),
            "nmi_max": partial(
                cmetrics.nmi, **{"y_true": y_true, "y_pred": y_pred, "average_method": "max"}
            ),
            "nmi_arithmetic": partial(
                cmetrics.nmi, **{"y_true": y_true, "y_pred": y_pred, "average_method": "arithmetic"}
            ),
            "euclidean_cluster_acc": partial(
                cmetrics.cluster_acc, **{"distances": edist, "y_true": y_true, "y_pred": y_pred}
            ),
            "cosine_cluster_acc": partial(
                cmetrics.cluster_acc, **{"distances": cdist, "y_true": y_true, "y_pred": y_pred}
            ),
        }

        for key, func in external_metrics_to_calculate.items():
            external_results[key] = self.__calculate_metrics_helper_(
                result_name=key, frozen_func=func
            )

        return external_results

    def __calculate_internal_metrics(
        self, edist: np.ndarray, cdist: np.ndarray, predicted_data: np.ndarray, y_pred: np.ndarray
    ) -> dict:
        internal_results = {}

        internal_metrics_to_calculate = {
            "euclidean_dunn1": partial(cmetrics.dunn, **{"distances": edist, "labels": y_pred}),
            "euclidean_dunn2": partial(cmetrics.dunn2, **{"distances": edist, "labels": y_pred}),
            "cosine_dunn1": partial(cmetrics.dunn, **{"distances": cdist, "labels": y_pred}),
            "cosine_dunn2": partial(cmetrics.dunn2, **{"distances": cdist, "labels": y_pred}),
            "euclidean_davies_bouldin": partial(
                cmetrics.davies_bouldin, **{"distances": edist, "labels": y_pred}
            ),
            "euclidean_davies_bouldin2": partial(
                cmetrics.davies_bouldin2, **{"distances": edist, "labels": y_pred}
            ),
            "euclidean_davies_bouldin3": partial(
                cmetrics.davies_bouldin3,
                **{"data": predicted_data, "labels": y_pred, "distance_func": euclidean_distances},
            ),
            "sklearn_davies_bouldin": partial(
                davies_bouldin_score,
                **{
                    "X": predicted_data,
                    "labels": y_pred,
                },
            ),
            "cosine_davies_bouldin": partial(
                cmetrics.davies_bouldin, **{"distances": cdist, "labels": y_pred}
            ),
            "cosine_davies_bouldin2": partial(
                cmetrics.davies_bouldin2, **{"distances": cdist, "labels": y_pred}
            ),
            "cosine_davies_bouldin3": partial(
                cmetrics.davies_bouldin3,
                **{"data": predicted_data, "labels": y_pred, "distance_func": cosine_distances},
            ),
            "euclidean_silhouette": partial(
                cmetrics.silhouette, **{"distances": edist, "y_pred": y_pred}
            ),
            "cosine_silhouette": partial(
                cmetrics.silhouette, **{"distances": cdist, "y_pred": y_pred}
            ),
        }

        for key, func in internal_metrics_to_calculate.items():
            internal_results[key] = self.__calculate_metrics_helper_(
                result_name=key, frozen_func=func
            )

        return internal_results

    def __calculate_metrics_(
        self,
        edist: np.ndarray,
        cdist: np.ndarray,
        representation_method_name: str,
        label_column: str,
        predicted_data: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        fit_params: dict = {},
        predict_params: dict = {},
    ) -> dict:
        results_dict = dict()

        results_dict["representation_name"] = representation_method_name
        results_dict["label_column"] = label_column
        results_dict["total_clusters"] = len(np.unique(y_pred))
        results_dict["representation_shape"] = predicted_data.shape

        internal_metrics_results = self.__calculate_internal_metrics(
            cdist=cdist, edist=edist, predicted_data=predicted_data, y_pred=y_pred
        )

        external_metrics_results = self.__calculate_external_metrics_(
            cdist=cdist, edist=edist, label_column=label_column, y_pred=y_pred, y_true=y_true
        )

        results_dict = {**results_dict, **internal_metrics_results, **external_metrics_results}
        results_dict["fit_params"] = str(fit_params)
        results_dict["predict_params"] = str(predict_params)

        return results_dict

    def __calculate_metrics_helper_(
        self, result_name: str, frozen_func: Callable[[], float]
    ) -> float:
        result = np.nan

        try:
            start_calculate = time()
            result = frozen_func()

            self.__logger_.debug(
                "calculating metric %s took %0.5f second(s)", result_name, time() - start_calculate
            )

        except Exception as calc_error:
            result = np.nan
            self.__logger_.error(
                "skipping metric %s (func %s) due to error: %s",
                result_name,
                frozen_func.func.__name__,
                str(calc_error),
            )
        return result

    def save_results(
        self,
        output_folder: str,
        metric_results: dict,
        result_file_name: str,
        representation_model_dict: dict,
    ) -> None:
        final_metric_results = {
            **metric_results,
            "representation_params": representation_model_dict,
        }
        full_path = os.path.join(output_folder, f"{result_file_name}.csv")
        open_mode = os.path.exists(full_path) and "a+" or "w+"
        results_start = time()

        with open(full_path, mode=open_mode, encoding="utf-8", newline="") as result_file:
            self.__logger_.debug(
                "opening file '%s' with mode '%s' took %0.5f second(s)",
                full_path,
                open_mode,
                time() - results_start,
            )

            writer = csv.DictWriter(f=result_file, fieldnames=final_metric_results.keys())

            if open_mode == "w+":
                results_start = time()

                writer.writeheader()

                self.__logger_.debug(
                    "writing header %s to file '%s' took %0.5f second(s)",
                    str(final_metric_results.keys()),
                    full_path,
                    time() - results_start,
                )

            results_start = time()
            writer.writerow(final_metric_results)

            self.__logger_.debug(
                "writing data %s to file '%s' with mode '%s' took %0.5f second(s)",
                str(final_metric_results),
                full_path,
                open_mode,
                time() - results_start,
            )

    def save_model(
        self,
        ainet_model: AiNet,
        representation_method_name: str,
        output_folder: str,
        result_filename: str,
        cur_iter: int,
        total_iter: int,
    ) -> None:
        save_start = time()

        final_filename=f"ainet_{result_filename}_{representation_method_name}_{cur_iter + 1}_{total_iter}"  # noqa: E501

        if os.path.exists(final_filename):
            final_filename += "_1"

        ainet_model.save_model(
            output_folder,
            filename=final_filename,
        )

        self.__logger_.debug(
            "exporting aiNet model for representation %s to file '%s' (%d of %d) took %0.5f second(s)",  # noqa: E501
            representation_method_name,
            os.path.join(output_folder, result_filename),
            cur_iter,
            total_iter,
            time() - save_start,
        )

    def __execute_representation_(
        self,
        representation_model_dict: dict,
        dataset_dataframe: pd.DataFrame,
        text_column: str,
        label_columns: np.ndarray,
        default_n_executions: int,
        ainet_model: AiNet,
        ainet_fit_params: dict,
        ainet_predict_params: list[dict],
        result_file_name: str,
        output_folder: str,
    ) -> None:
        (representation_model_name,) = (list(representation_model_dict.keys())[0],)

        _, representation_output, representation_model_dict = self.__generate_representation_(
            representation_name=representation_model_name,
            representation_model_dict=representation_model_dict,
            dataset_dataframe=dataset_dataframe,
            text_column=text_column,
        )

        # normalize_output = representation_model.get("normalize_output", False)

        normed_representation = representation_output.astype(np.float64)

        if self.__normed_representations_.get(representation_model_name, False):
            execution_start = time()
            normed_representation = MinMaxScaler().fit_transform(normed_representation)

            self.__logger_.debug(
                "normed representation for representation model %s with shape (%d, %d) took %0.5f second(s)",  # noqa: E501
                representation_model_name,
                normed_representation.shape[0],
                normed_representation.shape[1],
                time() - execution_start,
            )

        n_executions = representation_model_dict.get(representation_model_name, {}).get(
            "n_executions", default_n_executions
        )

        dist_start = time()
        edist = euclidean_distances(normed_representation, normed_representation)

        self.__logger_.debug(
            "calculating euclidean distance between %d and %d objects took %0.5f second(s)",  # noqa: E501
            normed_representation.shape[0],
            normed_representation.shape[0],
            time() - dist_start,
        )

        dist_start = time()
        cdist = cosine_distances(normed_representation, normed_representation)

        self.__logger_.debug(
            "calculating cosine distance between %d and %d objects took %0.5f second(s)",  # noqa: E501
            normed_representation.shape[0],
            normed_representation.shape[0],
            time() - dist_start,
        )

        label_column = label_columns[0]

        y_trues = {}
        original_results = {}
        for label_column in label_columns:
            y_true = dataset_dataframe[label_column].to_numpy()
            y_trues[label_column] = y_true

            if len(original_results.keys()) == 0:
                original_results = self.__calculate_metrics_(
                    cdist=cdist,
                    edist=edist,
                    y_true=y_true,
                    y_pred=y_true,
                    predicted_data=normed_representation,
                    representation_method_name="original_base",
                    label_column=label_column,
                )
            else:
                external_results = self.__calculate_external_metrics_(
                    cdist=cdist,
                    edist=edist,
                    label_column=label_column,
                    y_pred=y_true,
                    y_true=y_true,
                )

                original_results = {**original_results, **external_results}

            self.save_results(
                metric_results=original_results,
                result_file_name=result_file_name,
                output_folder=output_folder,
                representation_model_dict={
                    **representation_model_dict,
                    "train_corpus": True \
                        if representation_model_dict.get("train_corpus", False) \
                        else False
                },
            )

        for i in range(n_executions):
            execution_start = time()
            try:
                self.__execute_representation_iter_(
                    ainet_model=ainet_model,
                    cdist=cdist,
                    edist=edist,
                    representation_model_name=representation_model_name,
                    y_trues=y_trues,
                    normed_representation=normed_representation,
                    ainet_fit_params=ainet_fit_params,
                    ainet_predict_params=ainet_predict_params,
                    result_file_name=result_file_name,
                    cur_iter=i,
                    total_iter=n_executions,
                    output_folder=output_folder,
                    representation_model_dict=representation_model_dict,
                )

            except Exception as iter_error:
                self.__logger_.error(
                    "skipping iter %d of %d for representation %s due to error: %s",
                    i + 1,
                    n_executions,
                    representation_model_name,
                    str(iter_error),
                )

            finally:
                total_time = time() - execution_start

                self.__logger_.debug(
                    "iter %d of %d for representation model %s with shape (%d, %d) took %0.5f second(s)",  # noqa: E501
                    i + 1,
                    n_executions,
                    representation_model_name,
                    normed_representation.shape[0],
                    normed_representation.shape[1],
                    total_time,
                )

                print_progress_bar(
                    i + 1,
                    n_executions,
                    f"Representation model: {representation_model_name}",
                    f"Last iteration took {total_time:0.5f} second(s)",
                )

    def __execute_representation_iter_(
        self,
        ainet_model: AiNet,
        cdist: np.ndarray,
        edist: np.ndarray,
        representation_model_name: str,
        normed_representation: np.ndarray,
        ainet_fit_params: dict,
        ainet_predict_params: list[dict],
        y_trues: dict,
        result_file_name: str,
        cur_iter: int,
        total_iter: int,
        output_folder: str,
        representation_model_dict: dict,
    ) -> None:
        self.__logger_.debug(
            "fitting aiNet for representation model %s (iter %d of %d) with shape (%d, %d) and parameters %s",  # noqa: E501
            representation_model_name,
            cur_iter + 1,
            total_iter,
            normed_representation.shape[0],
            normed_representation.shape[1],
            str(ainet_fit_params),
        )

        cur_iter_start = time()
        ainet_model.fit(normed_representation, **ainet_fit_params)

        self.__logger_.debug(
            "aiNet fit for representation model %s (iter %d of %d) with shape (%d, %d) took %0.5f second(s)",  # noqa: E501
            representation_model_name,
            cur_iter + 1,
            total_iter,
            normed_representation.shape[0],
            normed_representation.shape[1],
            time() - cur_iter_start,
        )

        for predict_params in ainet_predict_params:
            pred_start = time()

            self.__logger_.debug(
                "predicting aiNet for representation model %s (iter %d of %d) with shape (%d, %d) and parameters %s",  # noqa: E501
                representation_model_name,
                cur_iter + 1,
                total_iter,
                normed_representation.shape[0],
                normed_representation.shape[1],
                str(predict_params),
            )

            y_pred = ainet_model.predict(normed_representation, **predict_params)

            self.__logger_.debug(
                "aiNet predict for representation model %s and predict params: (%s) (iter %d of %d) with shape (%d, %d) took %0.5f second(s)",  # noqa: E501
                representation_model_name,
                str(predict_params),
                cur_iter + 1,
                total_iter,
                normed_representation.shape[0],
                normed_representation.shape[1],
                time() - pred_start,
            )

            general_results = {}

            for label_column, y_true in y_trues.items():
                if len(general_results.keys()) == 0:
                    general_results = self.__calculate_metrics_(
                        cdist=cdist,
                        edist=edist,
                        representation_method_name=representation_model_name,
                        label_column=label_column,
                        predicted_data=normed_representation,
                        y_pred=y_pred,
                        y_true=y_true,
                        fit_params=ainet_fit_params,
                        predict_params=predict_params,
                    )
                else:
                    iter_metrics_results = self.__calculate_external_metrics_(
                        cdist=cdist,
                        edist=edist,
                        label_column=label_column,
                        y_pred=y_pred,
                        y_true=y_true,
                    )

                    general_results = {
                        **general_results,
                        **iter_metrics_results,
                    }

                self.save_results(
                    metric_results=general_results,
                    result_file_name=result_file_name,
                    output_folder=output_folder,
                    representation_model_dict={
                        **representation_model_dict,
                        "train_corpus": True \
                            if representation_model_dict.get("train_corpus", False) \
                            else False
                    },
                )

        self.save_model(
            ainet_model=ainet_model,
            representation_method_name=representation_model_name,
            result_filename=result_file_name,
            cur_iter=cur_iter,
            total_iter=total_iter,
            output_folder=output_folder,
        )

        self.__logger_.debug(
            "generating results for representation model %s (iter %d of %d) with shape (%d, %d) took %0.5f second(s)",  # noqa: E501
            representation_model_name,
            cur_iter + 1,
            total_iter,
            normed_representation.shape[0],
            normed_representation.shape[1],
            time() - cur_iter_start,
        )

    def execute(
        self,
        executions_plans: list[dict],
        output_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), "out"),
    ) -> None:
        """
        Executes a list of execution plans.

        Each execution plan is a dictionary that contains the parameters for a single execution.
        The method iterates over the list of execution plans and executes each one in turn.

        Args:
            executions_plans (list[dict]): A list of execution plans. Each execution plan
            is a dictionary that contains the parameters for a single execution.
        """
        total_executions = len(executions_plans)

        for idx, execution_plan in enumerate(executions_plans, start=1):
            dataset_path: str = execution_plan.get("dataset_path", None)
            text_column: str = execution_plan.get("text_column", None)
            label_columns: str = execution_plan.get("label_columns", None)
            delimiter: str = execution_plan.get("delimiter", None)

            self.__logger_.info(
                "validating file '%s' of execution plan %d of %d",
                dataset_path,
                idx,
                total_executions,
            )
            try:
                dataset_dataframe = self.__read_file_(
                    dataset_path=dataset_path,
                    text_column=text_column,
                    label_column=label_columns,
                    delimiter=delimiter,
                )
            except Exception as ex:
                raise Exception(
                    f"failed to read file '{dataset_path}'",
                ) from ex

            self.__logger_.info(
                "validation successful for file '%s' of execution plan %d of %d",
                dataset_path,
                idx,
                total_executions,
            )

        for idx, execution_plan in enumerate(executions_plans, start=1):
            cur_execution_plan_start = time()

            try:
                self.__logger_.info("starting execution plan %d of %d", idx, total_executions)

                default_n_executions: int = execution_plan.get("default_executions", 1)

                dataset_path: str = execution_plan.get("dataset_path", None)
                text_column: str = execution_plan.get("text_column", None)
                label_columns: str = execution_plan.get("label_columns", None)
                delimiter: str = execution_plan.get("delimiter", None)

                dataset_dataframe = self.__read_file_(
                    dataset_path=dataset_path,
                    text_column=text_column,
                    label_column=label_columns,
                    delimiter=delimiter,
                )

                default_result_file_name = "_".join(
                    "_".join(dataset_path.split(os.path.sep)[-3:]).split(os.path.extsep)[0:-1]
                )

                if not os.path.exists(output_folder):
                    self.__logger_.debug("%s directory does not exist, creating it", output_folder)
                    os.makedirs(output_folder)

                default_result_file_name = (
                    f"{default_result_file_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )

                result_file_name = execution_plan.get("file_output_name", default_result_file_name)

                assert result_file_name is not None and len(
                    result_file_name
                ), "result_file_name cannot be None or ''"

                ainet_model = self.__instantiate_ainet_(execution_plan)
                ainet_fit_params: dict = execution_plan.get("fit_params", {})
                ainet_predict_params: list[dict] = execution_plan.get("predict_params", [{}])

                plan_representation_models: list = execution_plan.get("representation_models", [])

                total_representation_models = len(plan_representation_models)

                for cur_model_idx, representation_model_dict in enumerate(
                    plan_representation_models, start=1
                ):
                    cur_representation_start = time()

                    try:
                        self.__execute_representation_(
                            representation_model_dict=representation_model_dict,
                            dataset_dataframe=dataset_dataframe,
                            text_column=text_column,
                            label_columns=label_columns,  # type: ignore
                            default_n_executions=default_n_executions,
                            ainet_model=ainet_model,
                            ainet_fit_params=ainet_fit_params,
                            ainet_predict_params=ainet_predict_params,
                            output_folder=output_folder,
                            result_file_name=result_file_name,
                        )

                    except Exception as _:
                        self.__logger_.error(
                            "skipping representation %s of plan %d of %d due to error: %s",  # noqa: E501
                            representation_model_dict.keys(),
                            idx,
                            total_executions,
                            format_exc(),
                        )
                        continue

                    finally:
                        total_time = time() - cur_representation_start
                        print_progress_bar(
                            cur_model_idx,
                            total_representation_models,
                            f"Running model {representation_model_dict.keys()} ({cur_model_idx} of {total_representation_models})",  # noqa: E501
                            f"Last model took {total_time:0.5f} second(s)",
                        )

            except Exception as error:
                self.__logger_.error(
                    "skipping execution plan %d of %d due to error: %s",
                    idx,
                    total_executions,
                    str(error),
                )

                continue

            finally:
                total_plan_time = time() - cur_execution_plan_start
                self.__logger_.debug(
                    "plan execution %d of %d took %0.5f second(s)",
                    idx,
                    total_executions,
                    total_plan_time,
                )

                print_progress_bar(
                    idx,
                    total_executions,
                    f"Current execution plan: {idx} of {total_executions}",
                    f"Last plan took {total_plan_time:0.5f} second(s)",
                )
