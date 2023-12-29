"""
    SIA.py
    ~~~~~~
    This module contains the SIA class.
    ~~~~~~
    criadopor:
        Lucas Eid Fernandes
        Matheus Amendoeira Ferraria
        Samuel Kenji Gomes
        Vinicius Amendoeira Ferraria
    ~~~~~~
    versao:
        1.0.0
    ~~~~~~
    data:
        30/10/2021
    ~~~~~~
    utilizado para o desenvolvimento do TCC
"""
from math import floor
from multiprocessing import Pool, cpu_count
from typing import Any, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from sklearn.metrics import confusion_matrix

from representations import NGram, Representation


class SIA:  # type: ignore
    """
        Classe do Sistema Imunológico Artificial Implementado
    """
    __slots__ = [
        "__bow_",
        "__dbow_",
        "__dbow_0_",
        "__dbow_1_",
        "__sentences_",
        "__classification_",
        "__detectors_",
        "__fit_threshold_",
        "__predict_threshold_",
        "__random_generator_",
        "__crossover_points_",
        "__analyzer_",
        "__representation_model_"
    ]

    __bow_: np.ndarray
    __dbow_: np.ndarray
    __dbow_0_: np.ndarray
    __dbow_1_: np.ndarray
    __sentences_: np.ndarray
    __classification_: np.ndarray
    __detectors_: list[np.ndarray]
    __fit_threshold_: float
    __predict_threshold_: float
    __random_generator_: Any
    __crossover_points_: float
    __representation_model_: Representation

    def __init__(self,
                 representation_model: Representation,
                 fit_threshold: float = 0.85,
                 predict_threshold: float = 0.0,
                 random_seed: int = 777) -> None:
        """
            Inicialização do SIA
        """

        assert issubclass(type(representation_model), Representation)

        assert isinstance(fit_threshold, float)\
            and 0 <= fit_threshold <= 1, "fit_threshold precisa estar entre 0 e 1"

        assert isinstance(predict_threshold, float)\
            and 0 <= predict_threshold <= 1, "predict_threshold precisa estar entre 0 e 1"

        assert isinstance(random_seed, int)

        self.__representation_model_ = representation_model
        self.__fit_threshold_ = fit_threshold
        self.__predict_threshold_ = predict_threshold\
            if predict_threshold > 0.0 \
            else self.__fit_threshold_

        self.__random_generator_ = np.random.default_rng(
            seed=random_seed)  # type: ignore
        self.__detectors_ = []

    def __repr__(self) -> str:
        """
            Método de representação da instância
        """
        return f"""Sentences: {self.__sentences_.shape}\n
        Detectors: {len(self.__detectors_)}\n"""

    def __getstate__(self) -> dict:
        obj_dict = {slot: getattr(self, slot)
                    for slot in self.__slots__ if slot != "__analyzer__"}

        return dict(obj_dict)

    def __setstate__(self, state: dict) -> None:
        for slot, value in state.items():

            object.__setattr__(self, slot, value)

    @property
    def dbow(self) -> np.ndarray:
        """
            Retorna o dbow em forma de um dataframe
        """
        # dbow_df =
        # dbow_df.insert(0, "sentences", self.__sentences__)
        # dbow_df.insert(1, "classification", self.__classification__)
        # return dbow_df
        return self.__representation_model_.representation

    @property
    def detectors(self) -> list[np.ndarray]:
        """
            Retorna os detetores
        """
        return self.__detectors_.copy()

    @property
    def sentences(self) -> Union[np.ndarray, list[Any]]:
        """
            Retorna uma cópia das frases utilizadas em treinamento
        """
        return self.__sentences_.tolist()

    @property
    def fit_threshold(self) -> float:
        """
            Retorna o threshold utilizado no treinamento
        """
        return self.__fit_threshold_

    @property
    def predict_threshold(self) -> float:
        """
            Retorna o threshold utilizando na classificação
        """
        return self.__predict_threshold_

    def __crossover_generation_(self) -> np.ndarray:
        """
            Gera os detetores utilizando a técnica de crossing-over
        """
        cell_size: int = self.__dbow_.shape[1]

        crossover_points: np.ndarray = self.__random_generator_.choice(
            range(1, cell_size + 1),
            size=self.__crossover_points_,
            replace=False
        )
        crossover_points = np.sort(crossover_points)

        if crossover_points.max() < cell_size:
            crossover_points = np.append(
                crossover_points, cell_size
            )  # type: ignore

        # seleciona os indices das linhas da matriz do dbow para cruzar
        index_parent_a: int = self.__random_generator_.choice(
            self.__dbow_0_.shape[0]
        )

        index_parent_b: int = self.__random_generator_.choice(
            self.__dbow_1_.shape[0]
        )

        noise_a = np.random.normal(0, 1, self.__dbow_0_.shape[1])
        vector_a: np.ndarray = self.__dbow_0_[index_parent_a] + noise_a

        noise_b = np.random.normal(0, 1, self.__dbow_1_.shape[1])
        vector_b: np.ndarray = self.__dbow_1_[index_parent_b] + noise_b

        candidate_detector: list[Any] = []
        last_index: int = 0

        for i, index in enumerate(crossover_points):
            if i % 2 == 0:
                candidate_detector = [
                    *candidate_detector,
                    *vector_a[last_index: index]
                ]
            else:
                candidate_detector = [
                    *candidate_detector,
                    *vector_b[last_index: index]
                ]

            last_index = index

        return np.array(candidate_detector)

    def __generate_candidate_detector_(self) -> np.ndarray:
        """
            Método responsável por gerar um único candidato a detector
        """
        return self.__crossover_generation_()

    def __generate_detectors_(self, number_of_detectors: int) -> None:
        """
            Método responsável por gerar os n detectores
        """
        self.__detectors_ = []
        number_of_processes = cpu_count() if number_of_detectors > 1000 else 1

        number_of_detectors_per_process = [
            (number_of_detectors//number_of_processes) for _ in range(number_of_processes)
        ]

        if number_of_detectors % number_of_processes:
            number_of_detectors_per_process[-1] += (
                number_of_detectors % number_of_processes
            )

        with Pool(processes=number_of_processes) as process_pool:
            results = process_pool.map(
                self.__generate_detectors_helper_,
                number_of_detectors_per_process, chunksize=1
            )

            for result in results:
                self.__detectors_ = [*self.__detectors_, *result]

    def __generate_detectors_helper_(self, number_of_detectors: int) -> list[np.ndarray]:
        """
            Método de apoio para permitir a geração em paralelo dos detectores
        """
        detectors = []
        # loop para criacao de detectores
        while len(detectors) < number_of_detectors:
            # gera um candidato aleatorio a ser um detector
            candidate_detector: np.ndarray = self.__generate_candidate_detector_()
            for d_row in self.__dbow_1_:  # type: ignore
                # verifica se o candidato a detector dá match com alguma linha do dbow (self)
                if self.__match_(
                    match_set=d_row,
                    detector=candidate_detector,
                    threshold=self.__fit_threshold_
                ):
                    break
            else:
                detectors.append(candidate_detector)
        return detectors

    def __cosine_similarity_(self, vector_a: np.ndarray, vector_b: np.ndarray) -> float:
        """
            Método responsável por calcular a similaridade entre dois vetores
        """
        top: float = (vector_a @ vector_b).sum()
        a_value: float = (np.sqrt(np.power(vector_a, 2).sum()))  # type: ignore
        b_value: float = (np.sqrt(np.power(vector_b, 2).sum()))  # type: ignore
        bottom: float = a_value * b_value

        if (not top or not bottom):
            return 0.0
        return top / bottom

    def __match_(self, match_set: np.ndarray, detector: np.ndarray, threshold: float) -> bool:
        """
            Método responsável por avaliar o quão similar os vetores são
        """
        return round(self.__cosine_similarity_(match_set, detector), 2) > threshold

    def fit(self, sentences: np.ndarray, classification: np.ndarray,
            number_of_detectors: int, crossover_points: float = 0.15) -> None:
        """
            Método responsável por treinar o modelo
        """

        assert isinstance(sentences, np.ndarray) \
            and sentences.dtype.type in [np.str_, np.object_]

        assert isinstance(classification, np.ndarray) \
            and classification.dtype.type in [np.int64, np.object_]

        assert isinstance(number_of_detectors, int) \
            and number_of_detectors >= 0, "number_of_detectors precisa ser maior que 0"

        assert isinstance(crossover_points, float) \
            and crossover_points >= 0, "crossover_points precisa ser maior que 0"

        self.__sentences_ = sentences
        self.__classification_ = classification
        self.__dbow_, self.__bow_ = self.__representation_model_.generate_representation(
            sentences
        )

        self.__crossover_points_ = crossover_points if crossover_points > 1\
            else floor(crossover_points * self.__dbow_.shape[1])

        # dbow com frases classificadas nao proprias
        self.__dbow_0_ = self.__dbow_[np.where(self.__classification_ == 0)]
        # dbow com frases classificadas como proprias
        self.__dbow_1_ = self.__dbow_[np.where(self.__classification_ == 1)]

        self.__generate_detectors_(number_of_detectors)

    def predict(self, sentences: np.ndarray) -> np.ndarray:
        """
            A partir da criaçãod de um dbow das sentencas a ser previstas
            o modelo realizar a classificação das sentenças recebidas
        """

        classification_results = []
        detect_dbow: np.ndarray

        detect_dbow, _ = self.__representation_model_.generate_representation(
            sentences=sentences, vocabulary=self.__bow_
        )

        number_of_processes = cpu_count() if len(self.__detectors_) * \
            len(sentences) > 1000 else 1

        partial_detect_dbows_indexes = np.array(
            [(0, detect_dbow.shape[0]//number_of_processes)
             for _ in range(number_of_processes)]
        )

        partial_detect_dbows_indexes = partial_detect_dbows_indexes.cumsum() \
            .reshape(number_of_processes, 2)

        if detect_dbow.shape[0] % number_of_processes:
            partial_detect_dbows_indexes[-1][-1] += (
                detect_dbow.shape[0] % number_of_processes)

        with Pool(processes=number_of_processes) as process_pool:
            results = process_pool.map(
                self.__predict_helper_,
                [detect_dbow[start:end]
                    for (start, end) in partial_detect_dbows_indexes],
                chunksize=1
            )

            for result in results:
                classification_results = [*classification_results, *result]
        return np.array(classification_results, int)

    def __predict_helper_(self, partial_detect_dbow: np.ndarray) -> np.ndarray:
        partial_classification = np.ones(
            partial_detect_dbow.shape[0], dtype=int)

        for index, row in enumerate(partial_detect_dbow):
            for detector in self.__detectors_:
                # verifica se o detector dá match com alguma linha do dbow
                if self.__match_(
                    match_set=row, detector=detector,
                    threshold=self.__predict_threshold_
                ):
                    partial_classification[index] = 0
        return partial_classification

    def build_confusion_matrix(self, classification: np.ndarray,
                               predictions: np.ndarray) -> np.ndarray:
        """
            Método responsável por gerar a matriz de confusão
        """
        return confusion_matrix(y_true=classification, y_pred=predictions)
        # confusion_matrix = DataFrame(
        #     columns=[0, 1], index=[0, 1], dtype=float)
        # confusion_matrix[0][0] = np.where(predictions[np.where(
        #     classification == 0)] == 0, 1, 0).sum()  # type: ignore
        # confusion_matrix[1][0] = np.where(predictions[np.where(
        #     classification == 0)] == 1, 1, 0).sum()  # type: ignore
        # confusion_matrix[0][1] = np.where(predictions[np.where(
        #     classification == 1)] == 0, 1, 0).sum()  # type: ignore
        # confusion_matrix[1][1] = np.where(predictions[np.where(
        #     classification == 1)] == 1, 1, 0).sum()  # type: ignore
        # return confusion_matrix

    def plot_confusion_matrix(self, nrows: int, ncols: int, sharex: bool, sharey: bool,
                              titles: Union[list[str], list[list[str]]], fig_title: str,
                              matrixes: DataFrame,
                              fig_size: tuple[float, float] = (20, 6))\
            -> tuple[object, list[list[plt.Axes]]]:
        # TODO change object return
        """
            Método responsável por plotar várias matrizes de confusão
        """
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey)
        fig.set_size_inches(*fig_size)
        fig.suptitle(fig_title)
        if nrows > 1:
            for row_index, row in enumerate(axes):  # type: ignore
                for col_index, col in enumerate(row):
                    show_cbar = (col_index == len(row)-1)
                    self.__plot_confusion_matrix_helper_(
                        plot_ax=col, title=titles[row_index][col_index],
                        # type: ignore
                        matrix_df=matrixes[row_index][col_index],
                        show_cbar=show_cbar
                    )
        else:
            for index, row_ax in enumerate(axes):  # type: ignore
                show_cbar = (index == len(axes)-1)  # type: ignore
                self.__plot_confusion_matrix_helper_(
                    plot_ax=row_ax, title=titles[index],  # type: ignore
                    # type: ignore
                    matrix_df=matrixes[index], show_cbar=show_cbar
                )
            # group_names = ["TN", "FP", "FN", "TP"]
        plt.show()
        return (fig, axes)

    def __plot_confusion_matrix_helper_(self, plot_ax: plt.Axes, title: str, matrix_df: Series,
                                        show_cbar: bool) -> None:
        """
            Método de apoio para plotar uma única matriz de confusão
        """
        plot_ax.set_title(title)
        sns.heatmap(matrix_df, annot=True, ax=plot_ax, fmt="0.0f", cmap="Blues",
                    annot_kws={"size": 18, "color": "black"}, cbar=show_cbar)

    # def export(self, file_path=os.getcwd()) -> NoReturn:
    #     with open(os.path.abspath(os.path.join(file_path, "dbow.txt")), "w+") as f:
    #         f.write(str(list(self._dbow)))


if __name__ == "__main__":
    chandelier = SIA(representation_model=NGram(), fit_threshold=0.65)
