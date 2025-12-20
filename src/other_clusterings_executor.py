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
from sklearn.cluster import (KMeans, DBSCAN, HDBSCAN)
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import davies_bouldin_score
from torch import cuda

import ainet.utils.evaluations.clustering_metrics as cmetrics
from ainet.utils import cosine_distances, euclidean_distances, print_progress_bar


class OtherClustersExecutor:
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

	def __init__(
		self, 
		logger: lg.Logger
	) -> None:
		init_start = time()
		self.__logger_ = logger

		assert logger is not None, "logger cannot be None"

		self.__logger_.debug(
			"Instantiating {%s} took %0.5f second(s)", self.__class__.__name__, time() - init_start
		)

	def __instantiate_cluster_model_(
		self, 
		model_config: dict[str, Any]
	) -> DBSCAN | KMeans | HDBSCAN:
		assert model_config is not None, "model_config cannot be None"
		
		klass_name = model_config.get("klass", None)
		
		params = model_config.get("params", {})
		assert klass_name is not None and len(klass_name), "klass_name cannot be None"
		assert isinstance(params, dict), "params must be a dict"
		
		model: DBSCAN | KMeans | HDBSCAN
		self. __logger_.debug("instantiating clustering model '%s' with params %s", klass_name, params)

		start_time = time()
  
		if klass_name == "HDBSCAN":
			model = HDBSCAN(**params)
			
		if klass_name == "KMeans":
			model = KMeans(**params)

		elif klass_name == "DBSCAN":
			model = DBSCAN(**params)

		self. __logger_.info("instantiated clustering model '%s' with params %s took %0.5f second(s)", klass_name, params, time() - start_time)

		return model

	def __read_file_(
		self, 
		dataset_path: str, 
		text_column: str, 
		label_column: str, 
		delimiter: str
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

	def __read_representation_(
		self, 
		name: str, 
		klass: str,
		parquet_path: str
	) -> pd.DataFrame:
		"""
		Load embeddings from a parquet file.

		Args:
			parquet_path: Path to the parquet file

		Returns:
			Tuple of (embedding_matrix, metadata)
		"""
  
		assert parquet_path is not None and len(parquet_path), "parquet_path cannot be None"
		
		abs_dataset_path = os.path.abspath(parquet_path)

		assert os.path.exists(abs_dataset_path), f"file '{abs_dataset_path}' does not exist"
		read_file_start = time()

		assert name is not None and len(name), "name cannot be None"
		assert klass is not None and len(klass), "klass cannot be None"

		# Read parquet file
		df = pd.read_parquet(parquet_path)

		# Extract embedding matrix (all columns starting with 'feat_')]
		feature_cols = [col for col in df.columns if not col.startswith("$meta_") and not col in ["$dataset", "$representation"]]
		embeddings = df[feature_cols]

		# Extract metadata
		metadata = {
			"dataset": df["$dataset"].iloc[0] if "$dataset" in df.columns else None,
			"representation": df["$representation"].iloc[0] if "$representation" in df.columns else None,
			"shape": embeddings.shape,
			"num_samples": embeddings.shape[0],
			"embedding_dim": embeddings.shape[1] if len(embeddings.shape) > 1 else 0,
		}

		# Extract any additional metadata columns
		meta_cols = [col for col in df.columns if col.startswith("$") or col in ["$dataset", "$representation"]]
		for col in meta_cols:
			metadata[col.replace("meta_", "")] = df[col].iloc[0]

		self.__logger_.info(
				"finished reading parquet '%s' containing representation %s[%s] with %d feature(s) and %d meta-column(s) took %0.5f second(s)",
				abs_dataset_path,
				name,
				klass,
                len(feature_cols),
                len(meta_cols),
				time() - read_file_start,
			)

		return embeddings, metadata

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
		self, 
		edist: np.ndarray, 
		cdist: np.ndarray, 
		predicted_data: np.ndarray, 
		y_pred: np.ndarray
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
        klass_name,
		klass_params: dict = {}
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
		results_dict["klass_name"] = klass_name
		results_dict["klass_params"] = str(klass_params)

		return results_dict

	def __calculate_metrics_helper_(
		self, 
		result_name: str, 
		frozen_func: Callable[[], float]
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

	def __save_results_(
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
			
	def __execute_representation_(
		self,
		representation_model_dict: dict,
		dataset_dataframe: pd.DataFrame,
		text_column: str,
		label_columns: np.ndarray,
		default_n_executions: int,
		model: DBSCAN | KMeans | HDBSCAN,
        klass_name: str,
		klass_params: dict,
		result_file_name: str,
		output_folder: str,
	) -> None:
		representation_model_name = representation_model_dict.get("name", None)
		representation_model_klass = representation_model_dict.get("klass", None)
		representation_parquet_path = representation_model_dict.get("parquet_path", None)

		assert representation_model_name is not None and len(representation_model_name), "representation_model_name cannot be None"  # noqa: E501
		assert representation_model_klass is not None and len(representation_model_klass), "representation_model_klass cannot be None"
		assert representation_parquet_path is not None and len(representation_parquet_path), "representation_parquet_path cannot be None"  # noqa: E501
		assert os.path.exists(representation_parquet_path), f"parquet_file '{representation_parquet_path}' does not exist"  # noqa: E501

		representation_output, _ = self.__read_representation_(
			name=representation_model_name,
			klass=representation_model_klass,
			parquet_path=representation_parquet_path
		)
		
		normed_representation = representation_output.astype(np.float64)

		if self.__normed_representations_.get(representation_model_klass, False):
			execution_start = time()
			normed_representation = MinMaxScaler().fit_transform(normed_representation)

			self.__logger_.debug(
				"normed representation for representation model %s[%s] with shape (%d, %d) took %0.5f second(s)",  # noqa: E501
				representation_model_name,
				representation_model_klass,
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
					edist=edist,
					cdist=cdist,
					representation_method_name="original_base",
					label_column=label_column,
					predicted_data=normed_representation,
					y_true=y_true,
					y_pred=y_true,
                    klass_name="Original"
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

			self.__save_results_(
				output_folder=output_folder,
				metric_results=original_results,
				result_file_name=result_file_name,
				representation_model_dict=representation_model_dict,
			)

		for i in range(n_executions):
			execution_start = time()
			try:
				self.__execute_representation_iter_(
					model=model,
					cdist=cdist,
					edist=edist,
					representation_model_name=representation_model_name,
					normed_representation=normed_representation,
                    klass_name=klass_name,
					klass_params=klass_params,
					y_trues=y_trues,
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
		model: DBSCAN | KMeans | HDBSCAN,
		cdist: np.ndarray,
		edist: np.ndarray,
		representation_model_name: str,
		normed_representation: np.ndarray,
		klass_name: str,
		klass_params: dict,
		y_trues: dict,
		result_file_name: str,
		cur_iter: int,
		total_iter: int,
		output_folder: str,
		representation_model_dict: dict,
	) -> None:
		pred_start = time()

		self.__logger_.debug(
			"fit_predicting %s for representation model %s (iter %d of %d) with shape (%d, %d) and parameters %s",  # noqa: E501
			type(model).__name__,
			representation_model_name,
			cur_iter + 1,
			total_iter,
			normed_representation.shape[0],
			normed_representation.shape[1],
			str(klass_params),
		)

		model.fit(normed_representation)
		y_pred = model.labels_

		self.__logger_.debug(
			"%s fit_predicting for representation model %s and predict params: (%s) (iter %d of %d) with shape (%d, %d) took %0.5f second(s)",  # noqa: E501
			type(model).__name__,
			representation_model_name,
			str(klass_params),
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
					y_true=y_true,
					y_pred=y_pred,
                    klass_name=klass_name, 
					klass_params=klass_params,
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

			self.__save_results_(
				metric_results=general_results,
				result_file_name=result_file_name,
				output_folder=output_folder,
				representation_model_dict=representation_model_dict
			)

		self.__logger_.debug(
			"generating results for representation model %s (iter %d of %d) with shape (%d, %d) took %0.5f second(s)",  # noqa: E501
			representation_model_name,
			cur_iter + 1,
			total_iter,
			normed_representation.shape[0],
			normed_representation.shape[1],
			time() - pred_start,
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
			dataset_config: dict = execution_plan.get("dataset", {})
			dataset_path: str = dataset_config.get("path", None)
			text_column: str = dataset_config.get("text_column", None)
			label_columns: str = dataset_config.get("label_columns", None)
			delimiter: str = dataset_config.get("delimiter", None)
			
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

				dataset_config: dict = execution_plan.get("dataset", {})
				dataset_path: str = dataset_config.get("path", None)
				text_column: str = dataset_config.get("text_column", None)
				label_columns: str = dataset_config.get("label_columns", None)
				delimiter: str = dataset_config.get("delimiter", None)

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

				models_config = execution_plan.get("models", [])
				
				for model_config in models_config:
					self.__logger_.info(
						"instantiating clustering model %s with params %s for execution plan %d of %d", 
						model_config.get("name", "unknown"),
						str(model_config.get("params", {})),
						idx, total_executions
					)
					model = self.__instantiate_cluster_model_(model_config)
					klass_name: str = model_config.get("klass", "unknown")
					klass_params: dict = model_config.get("params", {})
					plan_representation_models: list = execution_plan.get("representations", [])

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
								model=model,
                                klass_name=klass_name,
								klass_params=klass_params,
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
	
	


if __name__ == "__main__":
	## simple console logger setup
	root_log = lg.getLogger("root")
	
	## instantiate executor
	executor = OtherClustersExecutor(logger=root_log)

	## load new_tests_config.json
	import json
	with open("new_tests_config.json", "r", encoding="utf-8") as json_file:
		executions: dict = json.load(json_file)
		executor.execute(executions_plans=executions["plans"])
		