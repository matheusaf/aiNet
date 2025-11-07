"""
    aiNet
"""

import os
import pickle
from collections.abc import Callable
from ctypes import c_double
from functools import partial
from logging import Logger, getLogger
from multiprocessing import Array, Queue, cpu_count, get_context
from typing import Any

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from pandas import unique as pd_unique
from scipy.sparse.csgraph import minimum_spanning_tree

from utils import cosine_distances, euclidean_distances, print_progress_bar

MULTIPROCESSING_ARRAY_TYPE = c_double
ARRAY_DTYPE = np.float64


def _init_pool_process(
    shared_antigen_population: Any,
    random_generators: list[np.random.Generator],
    id_queue: Queue,
) -> None:
    global global_shared_antigen_population, global_random_generators, id_random_generator
    id_random_generator = id_queue.get()
    global_shared_antigen_population = shared_antigen_population
    global_random_generators = random_generators


def _convert_mp_to_nparray(
    mp_arr: Any,
    n_rows: int,
    n_cols: int,
    dtype: Any
) -> np.ndarray:
    return np.frombuffer(mp_arr.get_obj(), dtype=dtype)\
        .reshape(n_rows, n_cols)


class AiNet:
    """
        Ph.D. Thesis
        Leandro Nunes de Castro
        February, 2000
        Artificial Immune Network (aiNet) - Description in aiNet.doc
        Data normalization over [0,1] required
        Obs.: for simplicity of comprehension we chose non-complementary vectors
        Further extension: complementary vectors
        Secondary Functions: RUN_INET, DIST, CENTROID
        Internal functions: CLONE, SUPPRESS, VER_EQ, EXTRACT, PLOTVET1, DRAW_NET, NORMA
    """

    __slots__ = [
        "__logger_",
        "__antibody_population_",
        "__antigen_population_",
        "__antibody_labels_",
        "__antibody_distances_",
        "__mst_edges_deleted_",
        "__distance_func_",
        # TODO APENAS VISUALIZACAO
        "__general_random_generator_",
        "__antibody_population_history_",
        "__seed_sequence_",
        "__average_distance_per_iter_",
        "__using_multiprocessing_",
    ]

    __logger_: Logger
    __seed_sequence_: np.random.SeedSequence
    __antigen_population_: np.ndarray
    __antibody_labels_: np.ndarray
    __antibody_distances_: np.ndarray
    __antibody_population_: np.ndarray
    __general_random_generator_: np.random.Generator
    __mst_edges_deleted_: np.ndarray
    __distance_func_: Callable
    __using_multiprocessing_: bool

    # TODO APENAS VISUALIZACAO
    __antibody_population_history_: list
    __average_distance_per_iter_: list

    __valid_distance_methods_ = [
        "euclidean",
        "cosine"
    ]

    def __init__(self, logger: Logger, seed=777, distance_method: str = "euclidean") -> None:
        self.__logger_ = logger
        self.__seed_sequence_ = np.random.SeedSequence(entropy=seed)
        self.__antigen_population_ = np.array([], dtype=ARRAY_DTYPE)
        self.__antibody_population_ = np.array([], dtype=ARRAY_DTYPE)
        self.__mst_edges_deleted_ = np.ndarray([])
        self.__antibody_labels_ = np.array([], dtype=np.int8)
        self.__general_random_generator_ = np.random.default_rng(
            self.__seed_sequence_.spawn(1)[0]
        )

        assert distance_method.casefold() in self.__valid_distance_methods_, \
            f"distance_method must be one of the following values {self.__valid_distance_methods_}"

        self.__distance_func_ = euclidean_distances if distance_method == "euclidean" \
            else cosine_distances

    @property
    def antigen_population(self) -> np.ndarray:
        """
            get antigen population
            returns: current antigen population
        """
        return self.__antigen_population_.copy()

    @property
    def antibody_distances(self) -> np.ndarray:
        """
            get antigen population intra-distance
        """
        return self.__antibody_distances_.copy()

    @property
    def antibody_population(self) -> np.ndarray:
        """
            get antibody population
            returns: current antibody population
        """
        return self.__antibody_population_.copy()

    @property
    def mst_edges_deleted(self) -> np.ndarray:
        "mst_edges_deleted getter"
        return self.__mst_edges_deleted_.copy()

    @property
    def antibody_labels(self) -> np.ndarray:
        """antibody labels getter"""
        return self.__antibody_labels_.copy()

    # TODO APENAS VISUALIZACAO
    @property
    def antibody_population_history(self) -> list:
        """antibody population history getter"""
        return [*self.__antibody_population_history_]

    # TODO APENAS VISUALIZACAO
    @property
    def average_distance_per_iter(self) -> list:
        """average distance per iter getter"""
        return [*self.__average_distance_per_iter_]

    def _antibody_maturation(
        self,
        selected_antigen_pack: tuple[int, ARRAY_DTYPE],
        no_best_cells_taken_each_selection: int,
        clone_multiplier: float,
        pruning_threshold: float,
        percent_clones_reselected: float,
        suppression_threshold: float,
        n_cols: int,
        n_rows: int,
        dtype: Any
    ) -> tuple[np.ndarray, float]:
        global global_random_generators, id_random_generator

        selected_antigen_idx, hypermutation_rate = selected_antigen_pack
        random_generator = global_random_generators[id_random_generator]

        selected_antigen = _convert_mp_to_nparray(
            global_shared_antigen_population,
            n_cols=n_cols,
            n_rows=n_rows,
            dtype=dtype
        )[selected_antigen_idx]

        # 1.1.1. Determine its affinity fi,j, i = 1,...,N, to all Ab_i.
        #  fi,j = 1/Di,j, i = 1,...,N: Di,j =||Abi −Agj ||, i=1,...,N
        # Ag-Ab Affinity
        ag_ab_dist = self.__distance_func_(
            selected_antigen,
            self.__antibody_population_,
            using_multiprocessing=self.__using_multiprocessing_
        ).reshape(-1)

        # 1.1.2. A subset Ab{n} composed of the n highest affinity antibodies
        #   (1/D or sorted(D)) is selected;
        n_dist_index = np.argsort(ag_ab_dist)[
            :no_best_cells_taken_each_selection
        ]

        # 1.1.3. The n selected antibodies are going to proliferate (clone)
        # proportionally to their antigenic affinity fi,j,
        # generating a set C of clones: the higher the affinity
        # the larger the clone size
        # for each of the n selected antibodies
        number_of_clones = np.round(
            clone_multiplier -
            (ag_ab_dist[n_dist_index] * clone_multiplier)
        ).astype(int)

        # Clone & Affinity Maturation
        clone_population, clone_hypermutation_rate = self.__clone_(
            distance_matrix=ag_ab_dist,
            hypermutation_rate=hypermutation_rate,
            indexes=n_dist_index,
            number_of_clones_array=number_of_clones,
            random_generator=random_generator  # type: ignore
        )

        # 1.1.4. The set C is submitted to a directed affinity maturation process (guided
        # mutation) generating a mutated set C*,
        # where each antibody k from C* will suffer a mutation
        #  with a rate αk inversely proportional
        # to the antigenic affinity fi,j of its parent antibody: the higher the affinity,
        #   the smaller the mutation rate:
        # Ck*=Ck +αk (Agj –Ck);αk ∝1/fi,j;k=1,...,Nc;i=1,...,N.

        # Mutation
        clone_population += clone_hypermutation_rate * \
            (selected_antigen - clone_population)

        # 1.1.5 Determine the affinity dk,j = 1/Dk,j among Agj and all the elements of C*
        # Re-Selection
        ag_ab_dist = self.__distance_func_(
            selected_antigen,
            clone_population,
            using_multiprocessing=self.__using_multiprocessing_
        ).reshape(-1)

        n_dist_index = np.argsort(ag_ab_dist)

        # 1.1.6. From C*, re-select ζ% of the antibodies with highest dk,j
        #   and put them into a matrix Mj of clonal memory;
        number_of_reselected = np.ceil(
            percent_clones_reselected *
            clone_population.shape[0]
        ).astype(int)

        selected_idxs = n_dist_index[:number_of_reselected]

        cur_memory_matrix = clone_population[
            selected_idxs, :
        ]  # 1 clone for each Ag

        # 1.1.7. Apoptosis: eliminate all the memory clones from Mj
        # whose affinity Dk,j > σd:
        # Network Pruning (Natural Death)

        # new affinities
        cur_memory_dist = ag_ab_dist[selected_idxs]

        alive_indexes = self.__argpruning_(
            cur_memory_dist,
            pruning_threshold
        )[0]

        cur_memory_matrix = cur_memory_matrix[alive_indexes]

        cur_memory_dist = cur_memory_dist[alive_indexes]

        # 1.1.8. Determine the affinity si,k among the memory clones:
        # 1.1.9. Clonal suppression: eliminate those memory clones whose si,k < σs:
        cur_memory_matrix = self.__suppress_(
            cur_memory_matrix,
            suppression_threshold
        )

        best_distance = float(cur_memory_dist.min()) \
            if cur_memory_dist.size > 0 else -1.0

        return (cur_memory_matrix, best_distance)

    def fit(
        self,
        antigen_population: np.ndarray,
        number_of_antibodies: int,
        no_best_cells_taken_each_selection: int,
        clone_multiplier: float,
        pruning_threshold: float,
        percent_clones_reselected: float,
        suppression_threshold=0.001,
        hypermutation_rate=4.0,
        max_iter=100,
        stop_threshold=0.01,
        record_antibody_history=False
    ) -> None:
        """
            antigen_population -> antigens (training patterns)
            number_of_antibodies -> no. of antibodies (constructive)
            no_best_cells_taken_each_selection -> no. of best-matching cells taken
                for each Ag (Selection)
            clone_multiplier -> multiplier used to define the number of clones
                to be generated
            pruning_threshold -> natural death
            percent_clones_reselected -> percentile amount of clones to be Re-selected
            suppression_threshold -> suppression threshold (default: 0.001)
            hypermutation_rate -> learning (hypermutation) rate (default: 4.0)
            max_iter -> maximum number of iterations (default: 100)
            stop_threshold -> criteria used to determine if the population
                has converged to a optimal solution
        """
        hypermutation_rate = ARRAY_DTYPE(hypermutation_rate)
        self.__antibody_labels_ = np.array([])
        self.__mst_edges_deleted_ = np.array([])

        try:
            if not isinstance(antigen_population, np.ndarray):
                antigen_population = np.asarray(antigen_population)
        except Exception:
            self.__logger_.error(
                "error while converting to np array",
                exc_info=True
            )

        assert len(antigen_population.shape), \
            "antigen_population size must be greater then 0"

        assert (antigen_population.min() >= 0.0 and antigen_population.max() <= 1.1), \
            "populacao precisa estar entre 0 e 1"
        mp_context = get_context("fork")

        cur_iter = 1
        antigen_population_f64 = antigen_population.astype(ARRAY_DTYPE)
        antigens_population_size, feature_size = antigen_population_f64.shape
        shared_antigen_population = Array(
            MULTIPROCESSING_ARRAY_TYPE,
            antigens_population_size * feature_size
        )

        _convert_mp_to_nparray(
            shared_antigen_population,
            n_rows=antigens_population_size,
            n_cols=feature_size,
            dtype=MULTIPROCESSING_ARRAY_TYPE
        )[:] = antigen_population_f64

        self.__antibody_population_ = np.empty_like(  # pylint: disable=unexpected-keyword-arg
            antigen_population_f64,
            shape=(0, feature_size)
        )

        # TODO APENAS VISUALIZACAO
        self.__antibody_population_history_ = []

        self.__average_distance_per_iter_ = [1.0]

        number_of_processes = cpu_count() if antigens_population_size * \
            feature_size > 5_000 else 1
        chunksize = antigen_population_f64.shape[0] // number_of_processes

        self.__using_multiprocessing_ = number_of_processes > 1

        self.__logger_.info(
            "Using %d process(es) with chunksize of size %d",
            number_of_processes,
            chunksize
        )

        random_generators = [
            np.random.default_rng(seed)
            for seed in self.__seed_sequence_.spawn(number_of_processes)
        ]

        id_queue = Queue()

        for i in range(number_of_processes):
            id_queue.put(i)


        with mp_context.Pool(
            processes=number_of_processes,
            initializer=_init_pool_process,
            initargs=(shared_antigen_population, random_generators, id_queue)
        ) as process_pool:

            partial_antibody_maturation = partial(
                self._antibody_maturation,
                no_best_cells_taken_each_selection=no_best_cells_taken_each_selection,
                clone_multiplier=clone_multiplier,
                pruning_threshold=pruning_threshold,
                percent_clones_reselected=percent_clones_reselected,
                suppression_threshold=suppression_threshold,
                n_rows=antigens_population_size,
                n_cols=feature_size,
                dtype=MULTIPROCESSING_ARRAY_TYPE
            )

            while cur_iter <= max_iter \
                    and self.__average_distance_per_iter_[-1] > stop_threshold:

                self.__antibody_population_ = np.vstack(
                    (
                        self.__antibody_population_,
                        self.__generate_random_antibody_population_(
                            self.__general_random_generator_,
                            number_of_antibodies,
                            feature_size
                        )
                    )
                )

                antigen_idx_perm = self.__general_random_generator_.permutation(
                    antigens_population_size
                )

                hypermutation_rates = hypermutation_rate * \
                    (
                        ARRAY_DTYPE(0.9) **
                        np.arange(
                            0, antigens_population_size,
                            dtype=ARRAY_DTYPE
                        )
                    )

                # Utilizar uma funcao p/ controlar o valor de mi
                hypermutation_rate = hypermutation_rates[-1] * ARRAY_DTYPE(0.9)

                # 1.1. For each antigenic pattern Ag_j, j = 1,...,M, (Ag_j ∈ Ag), do:
                iterator_results = process_pool.imap_unordered(
                    partial_antibody_maturation,
                    iterable=zip(antigen_idx_perm, hypermutation_rates),
                    chunksize=chunksize
                )

                list_results = []

                best_distance_cum_sum, valid_distance_count = 0.0, 0.0

                # 1.1.10. Concatenate the total antibody memory matrix with the
                #  resultant clonal memory Mj* for Agj: Ab{m} ← [Ab{m};Mj*];
                for (iter_result, iter_best_distance) in iterator_results:
                    if iter_result.shape[0] > 0:
                        list_results.extend(iter_result)

                        if iter_best_distance > -1.0:
                            best_distance_cum_sum += iter_best_distance
                            valid_distance_count += 1

                if valid_distance_count == 0.0:
                    valid_distance_count = 1.0

                iter_mean_distance = float(best_distance_cum_sum) \
                    / float(valid_distance_count)

                memory_matrix = np.array(
                    list_results,
                    dtype=self.__antibody_population_.dtype,
                )

                # 1.2. Determine the affinity among all the memory antibodies from Ab{m}:
                # 1.3. Network suppression: eliminate all the antibodies such that si,k < σs:
                memory_matrix = self.__suppress_(
                    memory_matrix,
                    suppression_threshold
                )

                self.__average_distance_per_iter_.append(
                    iter_mean_distance
                )

                # 1.4. Build the total antibody matrix Ab ← [Ab{m};Ab{d}]
                self.__antibody_population_ = memory_matrix

                print_progress_bar(
                    cur_iter,
                    max_iter,
                    prefix=f"iter: {cur_iter} | cur_hyper_rate: {hypermutation_rate: .8f} |",
                    suffix=f"| avd: {self.__average_distance_per_iter_[-1]:.05f} | net size: {memory_matrix.shape[0]}"  # noqa: E501
                )

                if record_antibody_history:
                    # TODO APENAS VISUALIZACAO
                    self.__antibody_population_history_.append(
                        self.__antibody_population_
                    )

                cur_iter += 1

        self.__antibody_distances_ = self.__distance_func_(
            self.__antibody_population_,
            self.__antibody_population_,
            using_multiprocessing=self.__using_multiprocessing_
        )

        del shared_antigen_population
        self.__antigen_population_ = antigen_population_f64

        self.__logger_.info("Finished at iter %d", cur_iter - 1)

    def predict(
        self,
        predict_data: np.ndarray,
        mst_pruning_threshold=0.105,
        mst_pruning_type="threshold",
        k: float = 0.0,
        minimum_no_edges=1
    ) -> NDArray[np.int8]:
        """
            method to classify new data
        """
        self.__logger_.debug(
            "predicting objects with shape (%d, %d)",
            predict_data.shape[0],
            predict_data.shape[1],
        )

        antibody_dist = np.triu(self.__antibody_distances_)

        tree = minimum_spanning_tree(antibody_dist).toarray()

        graph: nx.Graph = nx.from_numpy_array(tree)

        self.__mst_pruning_(
            graph=graph,
            pruning_threshold=mst_pruning_threshold,
            pruning_type=mst_pruning_type,
            k=k,
            minimum_no_edges=minimum_no_edges
        )

        self.__antibody_labels_ = self.__generate_cluster_result_(graph)

        distance = self.__distance_func_(
            predict_data,
            self.__antibody_population_,
            using_multiprocessing=self.__using_multiprocessing_
        )

        if distance.size == 0:
            return np.ones_like(self.__antibody_labels_) * -1

        closest_ab_idx = np.argmin(distance, axis=1)

        self.__logger_.debug(
            "finished predicting objects with shape (%d, %d)",
            predict_data.shape[0],
            predict_data.shape[1],
        )

        return self.__antibody_labels_[closest_ab_idx]

    def __clone_(
        self,
        distance_matrix: np.ndarray,
        hypermutation_rate: ARRAY_DTYPE,
        indexes: np.ndarray,
        number_of_clones_array: np.ndarray,
        random_generator: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:

        number_of_columns = self.__antibody_population_.shape[1]
        total_clones = np.sum(number_of_clones_array).astype(np.int32)

        clone_population = np.empty_like(  # pylint: disable=unexpected-keyword-arg
            self.__antibody_population_,
            shape=(total_clones, number_of_columns)
        )

        clone_hyptermutation_rate = np.empty_like(  # pylint: disable=unexpected-keyword-arg
            self.__antibody_population_,
            shape=(total_clones, number_of_columns)
        )

        prev_clone_end_idx = 0

        for idx, number_of_clones in zip(indexes, number_of_clones_array):
            cur_clone_end_idx = prev_clone_end_idx + number_of_clones

            clone_population[prev_clone_end_idx: cur_clone_end_idx] = np.resize(
                self.__antibody_population_[idx, :],
                (number_of_clones, number_of_columns)
            )

            random_distribuition = self.__generate_random_antibody_population_(
                random_generator,
                number_of_clones,
                number_of_columns
            )

            clone_hyptermutation_rate[prev_clone_end_idx: cur_clone_end_idx] = \
                random_distribuition * \
                distance_matrix[idx] * hypermutation_rate

            prev_clone_end_idx = cur_clone_end_idx

        return (clone_population, clone_hyptermutation_rate)

    def __argpruning_(
        self,
        distance_matrix: np.ndarray,
        pruning_threshold: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
            @Input: distance_matrix: np.ndarray -> distance matrix used to prune
            @Input: pruning_threshold float -> pruning threshold

            Returns distance matrix indexes of surviving cells
        """

        indexes = np.where(
            distance_matrix <= pruning_threshold
        )

        return indexes  # type: ignore

    def __suppress_(
        self,
        memory_matrix: np.ndarray,
        suppression_threshold: float
    ) -> np.ndarray:
        if memory_matrix.size == 0:
            return memory_matrix

        matrix_dist = self.__distance_func_(
            memory_matrix,
            memory_matrix,
            using_multiprocessing=self.__using_multiprocessing_
        )

        matrix_dist = np.triu(matrix_dist) + \
            (np.tril(np.ones_like(matrix_dist, dtype=np.int8) * -1))

        alive_indexes = np.logical_not(
            np.logical_and(
                matrix_dist >= 0,
                matrix_dist < suppression_threshold
            )
        ).all(axis=1).nonzero()[0]

        if alive_indexes.size > 0:
            alive_indexes = np.array(pd_unique(alive_indexes))
            # 1.3. Network suppression: eliminate all the antibodies such that si,k < σs:
            # memory_matrix = extract(memory_matrix, Is)
            return memory_matrix[alive_indexes, :]

        return memory_matrix

    def __generate_random_antibody_population_(
        self,
        random_generator: np.random.Generator,
        population_size: int,
        feature_size: int
    ) -> np.ndarray:
        return random_generator.uniform(
            0, 1,
            size=(population_size, feature_size)
        ).astype(ARRAY_DTYPE)

    def __mst_pruning_(
        self,
        graph: nx.Graph,
        pruning_threshold: float,
        pruning_type: str,
        k: float,
        minimum_no_edges: int
    ) -> None:
        edges_to_be_deleted = []

        def handle_weigth_data_null(edge_weight_data) -> float:
            return 0.0 if edge_weight_data is None else edge_weight_data

        for (u_index, v_index, edge_weight) in graph.edges(data="weight"):  # type: ignore
            final_edge_weight = handle_weigth_data_null(edge_weight)
            should_be_deleted = False

            u_weights = [
                handle_weigth_data_null(e[-1])
                for e in graph.edges(
                    u_index, data="weight"  # type: ignore
                )
                if e[1] != v_index
            ]

            v_weights = [
                handle_weigth_data_null(e[-1])
                for e in graph.edges(
                    v_index, data="weight"  # type: ignore
                )
                if e[1] != u_index
            ]

            if len(u_weights) >= minimum_no_edges and len(v_weights) >= minimum_no_edges:
                u_mean = np.mean(u_weights)
                v_mean = np.mean(v_weights)

                if pruning_type == "average":
                    u_factor = u_mean + (np.std(u_weights) * k)
                    v_factor = v_mean + (np.std(v_weights) * k)
                    should_be_deleted = (final_edge_weight > u_factor) \
                        and (final_edge_weight > v_factor)

                elif pruning_type == "threshold":
                    should_be_deleted = u_mean > pruning_threshold and v_mean > pruning_threshold

                if should_be_deleted:
                    edges_to_be_deleted.append((u_index, v_index))

        if len(edges_to_be_deleted) > 0:
            graph.remove_edges_from(edges_to_be_deleted)
            self.__mst_edges_deleted_ = np.array(edges_to_be_deleted)

    def __generate_cluster_result_(
        self,
        graph: nx.Graph
    ) -> NDArray[np.int8]:
        total_nodes = graph.number_of_nodes()
        cluster_result = np.zeros(total_nodes, dtype=np.int8)
        visited = np.zeros(total_nodes).astype(bool)

        cluster_index = 0

        while not all(visited):
            cur_cluster_index = cluster_index
            # pylint: disable = singleton-comparison
            cur_index: int = np.where(np.logical_not(visited))[0][0]

            queue = [cur_index]

            visited[cur_index] = True
            cluster_result[cur_index] = cluster_index

            # running Breadth First Search (BFS) for each disconnected Graph
            # in order to build cluster
            while len(queue) > 0:
                cur_index = queue.pop(0)
                for _, v_index in graph.edges(cur_index):  # type: ignore
                    if not visited[v_index]:  # pylint: disable = singleton-comparison
                        queue.insert(0, v_index)
                        visited[v_index] = True
                        cluster_result[v_index] = cur_cluster_index
                        if cur_cluster_index == cluster_index:
                            cluster_index += 1

        return cluster_result

    def save_model(self, filepath: str,  filename: str) -> None:
        """Saves the current model into a pickle file"""
        final_filepath = os.path.join(filepath, f"{filename}.pkl")
        with open(final_filepath, "wb+") as file:
            pickle.dump(self,  file)


if __name__ == "__main__":
    import csv

    # from representations import STagger
    from sklearn.preprocessing import MinMaxScaler

    from utils.evaluations.clustering_metrics import (
        cluster_confusion_matrix,
        map_ypred_to_ytrue
    )

    with open(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "shared", "datasets", "ruspini.csv"),
            # "shared", "datasets", "Sentiment Labelled", "imdb_labelled.txt"),
        encoding="utf-8"
    ) as f:
        reader = csv.reader(f, delimiter=";")
        # reader = csv.reader(f, delimiter="\t")
        ruspini = [line for line in reader]
        headers = ruspini.pop(0)
        ruspini = np.array(ruspini, dtype=object)

    # data = STagger().generate_representation(ruspini[:, 0])[1]
    data = ruspini[:, 1:-1].astype(int)  # type: ignore
    target = ruspini[:, -1]  # type: ignore

    normed_data = MinMaxScaler().fit_transform(data[:, :2])
    # normed_data = MinMaxScaler().fit_transform(data) # type: ignore

    net = AiNet(logger=getLogger("root"))
    # net = AiNet(distance_method="cosine")

    net.fit(
        normed_data,
        number_of_antibodies=30,
        no_best_cells_taken_each_selection=4,
        clone_multiplier=5,
        pruning_threshold=0.8,
        percent_clones_reselected=0.12,
        suppression_threshold=0.1,
        max_iter=18,
        record_antibody_history=True
    )

    print(net.antibody_population.shape)

    groups = net.predict(normed_data, mst_pruning_type="average", k=1.5)

    dist = euclidean_distances(normed_data, normed_data)

    print(groups)
    confusion_matrix = cluster_confusion_matrix(
        target,
        map_ypred_to_ytrue(dist, target, groups)
    )
    print(confusion_matrix)
    print(confusion_matrix.diagonal().sum() / confusion_matrix.sum())
