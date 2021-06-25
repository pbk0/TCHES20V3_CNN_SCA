import sys
import zipfile

import pandas as pd

sys.path.append("..")
import typing as t
import pathlib
import numpy as np
import enum
from sklearn.model_selection import train_test_split
import tqdm
import plotly.graph_objects as go
import pickle
import plotly.express as px
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from keras.models import load_model

from src.clr import OneCycleLR
from src import dataLoaders
from src import preproces
from src import models

ROOT_DIR = pathlib.Path(__file__).parent / "results"
NUM_ATTACKS_PER_EXPERIMENT = 100
NUM_EXPERIMENTS = 100


def preprocess_predictions(predictions, all_guess_targets, num_examples, num_guesses) -> np.ndarray:

    # make copy
    predictions = predictions.copy()

    # Add small positive value
    # note that we set any o or negative probability to smallest
    # possible positive number so that np.log does not
    # result to -np.inf
    predictions[predictions <= 1e-45] = 1e-45

    # Sort based on guessed targets
    sorted_predictions = predictions[
        np.asarray(
            [np.arange(num_examples)]
        ).repeat(num_guesses, axis=0).T,
        all_guess_targets
    ]

    # take negative logs
    sorted_neg_log_preds = -np.log(sorted_predictions)

    # return
    return sorted_neg_log_preds


def compute_ranks(predictions, all_guess_targets, correct_key, num_attacks) -> np.ndarray:

    # num_examples and num_guesses
    num_examples = predictions.shape[0]
    num_guesses = 256

    # some buffers
    all_ranks = np.zeros((num_attacks, num_examples), np.uint8)

    # fix seed for deterministic behaviour
    np.random.seed(123456)

    # get sorted_neg_log_preds
    sorted_neg_log_preds = preprocess_predictions(predictions, all_guess_targets, num_examples, num_guesses)

    # loop over
    for attack_id in tqdm.trange(num_attacks):
        # first shuffle for simulating random experiment
        np.random.shuffle(sorted_neg_log_preds)

        # cum sum
        sorted_neg_log_preds_cum_sum = np.cumsum(sorted_neg_log_preds, axis=0)

        # compute rank
        ranks_for_all_guesses = sorted_neg_log_preds_cum_sum.argsort().argsort()

        # set correct rank
        all_ranks[attack_id, :] = ranks_for_all_guesses[:, correct_key]

    # return
    return all_ranks


class Dataset(enum.Enum):
    ascad_0 = enum.auto()
    ascad_50 = enum.auto()
    ascad_100 = enum.auto()
    aes_hd = enum.auto()
    aes_rd = enum.auto()
    dpav4 = enum.auto()

    @property
    def rank_plot_until(self) -> int:
        if self in [self.ascad_0, self.ascad_50, self.ascad_100, ]:
            return 250
        elif self is self.aes_hd:
            return 1200
        elif self is self.aes_rd:
            return 10
        elif self is self.dpav4:
            return 10
        else:
            raise Exception(f"Unsupported dataset `{self.name}`")

    def get(
        self
    ) -> t.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        # Load a dataset (see src/dataLoaders.py)
        if self is self.ascad_0:
            _data = dataLoaders.load_ascad(f'./../datasets/ASCAD_dataset/ASCAD.h5')
        elif self is self.ascad_50:
            _data = dataLoaders.load_ascad(f'./../datasets/ASCAD_dataset/ASCAD_desync50.h5')
        elif self is self.ascad_100:
            _data = dataLoaders.load_ascad(f'./../datasets/ASCAD_dataset/ASCAD_desync100.h5')
        elif self is self.aes_hd:
            _data = dataLoaders.load_aes_hd(f'./../datasets/AES_HD_dataset/')
        elif self is self.aes_rd:
            _data = dataLoaders.load_aes_rd(f'./../datasets/AES_RD_dataset/')
        elif self is self.dpav4:
            _data = dataLoaders.load_dpav4(f'./../datasets/DPAv4_dataset/')
        else:
            raise Exception(f"Dataset `{self}` is not supported")

        return _data


class Preprocessor(enum.Enum):
    none = enum.auto()
    feature_standardization = enum.auto()
    horizontal_standardization = enum.auto()

    @property
    def preprocess_fn(self) -> t.Callable:

        # get preprocessor
        if self is self.none:
            _preprocessor = preproces.no_preprocessing
        elif self is self.feature_standardization:
            _preprocessor = preproces.feature_standardization
        elif self is self.horizontal_standardization:
            _preprocessor = preproces.horizontal_standardization
        else:
            raise Exception(f"Preprocessor `{self}` is not supported")

        return _preprocessor

    def apply(
        self, dataset: Dataset
    ) -> t.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        # get dataset
        X_profiling, Y_profiling, X_attack, targets, key_attack = dataset.get()

        # Preprocess the data (see src/preproces.py)
        X_profiling_processed, X_attack_processed = self.preprocess_fn(X_profiling, X_attack)

        # Automatic split (90/10)
        # Random_state is set to get reproducible splits
        # model weights are initialised randomly (can also saved in case we encouter weird things)
        tracesTrain, tracesVal, labelsTrain, labelsVal = train_test_split(
            X_profiling_processed, Y_profiling, test_size=0.1, random_state=0)

        # return
        return tracesTrain, tracesVal, labelsTrain, labelsVal, X_attack_processed, targets, key_attack


class Model(enum.Enum):
    ascad_mlp = enum.auto()
    ascad_cnn = enum.auto()
    ascad_mlp_fn = enum.auto()
    ascad_cnn_fn = enum.auto()
    eff_cnn = enum.auto()
    simplified_eff_cnn = enum.auto()

    # noinspection DuplicatedCode
    def make_fn(self, dataset: Dataset) -> t.Callable:
        if self in [self.ascad_mlp, self.ascad_mlp_fn, ]:
            if dataset in [Dataset.ascad_0, Dataset.ascad_50, Dataset.ascad_100, ]:
                return models.ascad_mlp_best
        elif self in [self.ascad_cnn, self.ascad_cnn_fn, ]:
            if dataset in [Dataset.ascad_0, Dataset.ascad_50, Dataset.ascad_100, ]:
                return models.ascad_cnn_best
        elif self is self.eff_cnn:
            if dataset is Dataset.ascad_0:
                return models.zaid_ascad_desync_0
            elif dataset is Dataset.ascad_50:
                return models.zaid_ascad_desync_50
            elif dataset is Dataset.ascad_100:
                return models.zaid_ascad_desync_100
            elif dataset is Dataset.aes_hd:
                return models.zaid_aes_hd
            elif dataset is Dataset.aes_rd:
                return models.zaid_aes_rd
            elif dataset is Dataset.dpav4:
                return models.zaid_dpav4
        elif self is self.simplified_eff_cnn:
            if dataset is Dataset.ascad_0:
                return models.noConv1_ascad_desync_0
            elif dataset is Dataset.ascad_50:
                return models.noConv1_ascad_desync_50
            elif dataset is Dataset.ascad_100:
                return models.noConv1_ascad_desync_100
            elif dataset is Dataset.aes_hd:
                return models.noConv1_aes_hd
            elif dataset is Dataset.aes_rd:
                return models.noConv1_aes_rd
            elif dataset is Dataset.dpav4:
                return models.noConv1_dpav4
        else:
            raise Exception(f"Model {self.model} is not supported")
        raise Exception(
            f"`{dataset}` dataset cannot be used with `{self.model}` model"
        )


class Params(t.NamedTuple):
    epochs: int
    batch_size: int
    learning_rate: float
    one_cycle_lr: bool
    preprocessor: Preprocessor


DEFAULT_PARAMS = {
    Dataset.ascad_0: {
        Model.ascad_mlp: Params(
            epochs=200, batch_size=100, learning_rate=0.00001, one_cycle_lr=False,
            preprocessor=Preprocessor.none,
        ),
        # Model.ascad_cnn: Params(
        #     epochs=75, batch_size=200, learning_rate=0.00001, one_cycle_lr=False,
        #     preprocessor=Preprocessor.none,
        # ),
        Model.ascad_mlp_fn: Params(
            epochs=200, batch_size=100, learning_rate=0.00001, one_cycle_lr=False,
            preprocessor=Preprocessor.feature_standardization,
        ),
        # Model.ascad_cnn_fn: Params(
        #     epochs=75, batch_size=200, learning_rate=0.00001, one_cycle_lr=False,
        #     preprocessor=Preprocessor.feature_standardization,
        # ),
        Model.eff_cnn: Params(
            epochs=50, batch_size=50, learning_rate=5e-3, one_cycle_lr=True,
            preprocessor=Preprocessor.feature_standardization,
        ),
        Model.simplified_eff_cnn: Params(
            epochs=50, batch_size=50, learning_rate=5e-3, one_cycle_lr=True,
            preprocessor=Preprocessor.feature_standardization,
        ),
    },
    Dataset.ascad_50: {
        Model.ascad_mlp: Params(
            epochs=200, batch_size=100, learning_rate=0.00001, one_cycle_lr=False,
            preprocessor=Preprocessor.none,
        ),
        # Model.ascad_cnn: Params(
        #     epochs=75, batch_size=200, learning_rate=0.00001, one_cycle_lr=False,
        #     preprocessor=Preprocessor.none,
        # ),
        Model.ascad_mlp_fn: Params(
            epochs=200, batch_size=100, learning_rate=0.00001, one_cycle_lr=False,
            preprocessor=Preprocessor.horizontal_standardization,
        ),
        # Model.ascad_cnn_fn: Params(
        #     epochs=75, batch_size=200, learning_rate=0.00001, one_cycle_lr=False,
        #     preprocessor=Preprocessor.horizontal_standardization,
        # ),
        Model.eff_cnn: Params(
            epochs=50, batch_size=256, learning_rate=5e-3, one_cycle_lr=True,
            preprocessor=Preprocessor.horizontal_standardization,
        ),
        Model.simplified_eff_cnn: Params(
            epochs=50, batch_size=50, learning_rate=5e-3, one_cycle_lr=True,
            preprocessor=Preprocessor.horizontal_standardization,
        ),
    },
    Dataset.ascad_100: {
        Model.ascad_mlp: Params(
            epochs=200, batch_size=100, learning_rate=0.00001, one_cycle_lr=False,
            preprocessor=Preprocessor.none,
        ),
        # Model.ascad_cnn: Params(
        #     epochs=75, batch_size=200, learning_rate=0.00001, one_cycle_lr=False,
        #     preprocessor=Preprocessor.none,
        # ),
        Model.ascad_mlp_fn: Params(
            epochs=200, batch_size=100, learning_rate=0.00001, one_cycle_lr=False,
            preprocessor=Preprocessor.horizontal_standardization,
        ),
        # Model.ascad_cnn_fn: Params(
        #     epochs=75, batch_size=200, learning_rate=0.00001, one_cycle_lr=False,
        #     preprocessor=Preprocessor.horizontal_standardization,
        # ),
        Model.eff_cnn: Params(
            epochs=50, batch_size=256, learning_rate=1e-2, one_cycle_lr=True,
            preprocessor=Preprocessor.horizontal_standardization,
        ),
        Model.simplified_eff_cnn: Params(
            epochs=50, batch_size=50, learning_rate=5e-3, one_cycle_lr=True,
            preprocessor=Preprocessor.horizontal_standardization,
        ),
    },
    Dataset.aes_hd: {
        Model.eff_cnn: Params(
            epochs=20, batch_size=256, learning_rate=1e-3, one_cycle_lr=False,
            preprocessor=Preprocessor.feature_standardization,
        ),
        Model.simplified_eff_cnn: Params(
            epochs=20, batch_size=256, learning_rate=1e-3, one_cycle_lr=False,
            preprocessor=Preprocessor.feature_standardization,
        ),
    },
    Dataset.aes_rd: {
        Model.eff_cnn: Params(
            epochs=50, batch_size=50, learning_rate=10e-3, one_cycle_lr=True,
            preprocessor=Preprocessor.horizontal_standardization,
        ),
        Model.simplified_eff_cnn: Params(
            epochs=50, batch_size=50, learning_rate=10e-3, one_cycle_lr=True,
            preprocessor=Preprocessor.horizontal_standardization,
        ),
    },
    Dataset.dpav4: {
        Model.eff_cnn: Params(
            epochs=50, batch_size=50, learning_rate=1e-3, one_cycle_lr=False,
            preprocessor=Preprocessor.feature_standardization,
        ),
        Model.simplified_eff_cnn: Params(
            epochs=50, batch_size=50, learning_rate=1e-3, one_cycle_lr=False,
            preprocessor=Preprocessor.feature_standardization,
        ),
    },
}  # type: t.Dict[Dataset, t.Dict[Model, Params]]


class Experiment(t.NamedTuple):
    id: int
    dataset: Dataset
    model: Model
    early_stopping: bool

    @property
    def name(self) -> str:
        _es_no_es = "es" if self.early_stopping else "no_es"
        return f"{self.dataset.name}-{self.model.name}-{_es_no_es}[{self.id}]"

    @property
    def store_dir(self) -> pathlib.Path:
        _es_no_es = "es" if self.early_stopping else "no_es"
        _ret = ROOT_DIR / self.dataset.name / self.model.name / _es_no_es / str(self.id)
        if not _ret.exists():
            _ret.mkdir(parents=True)
        return _ret

    @property
    def model_file_path(self) -> pathlib.Path:
        return self.store_dir / "model.hdf5"

    @property
    def history_file_path(self) -> pathlib.Path:
        return self.store_dir / "history.pickle"

    @property
    def ranks_file_path(self) -> pathlib.Path:
        return self.store_dir / "ranks.npy"

    @property
    def is_executing_file_path(self) -> pathlib.Path:
        return self.store_dir / "__is_executing__"

    @property
    def is_executing(self) -> bool:
        return self.is_executing_file_path.exists()

    @property
    def is_done(self) -> bool:
        if self.is_executing:
            return False
        if self.model_file_path.exists() and self.history_file_path.exists() and self.ranks_file_path.exists():
            return True
        else:
            return False

    @property
    def default_params(self) -> Params:
        try:
            return DEFAULT_PARAMS[self.dataset][self.model]
        except KeyError:
            raise Exception(
                f"Default parameters not available for model `{self.model.name}` with dataset `{self.dataset.name}`"
            )

    @property
    def ranks(self) -> np.ndarray:
        # noinspection PyTypeChecker
        return np.load(self.ranks_file_path.resolve().as_posix())

    @property
    def losses(self) -> t.Tuple[np.ndarray, np.ndarray]:
        with open(self.history_file_path.as_posix(), 'rb') as file_pi:
            history = pickle.load(file_pi)
        train_loss = history['loss']
        val_loss = history['val_loss']
        return train_loss, val_loss

    def dump_plots(self):
        # ---------------------------------------------------- 01
        # check if done
        if not self.is_done:
            raise Exception(f"Cannot dump plots as the experiment is not completed ...")

        # ---------------------------------------------------- 02
        # name
        _name = f"{self.dataset.name}-{self.model.name}"

        # ---------------------------------------------------- 03
        # create figures
        avg_rank_fig = go.Figure(
            layout=go.Layout(
                title=go.layout.Title(text=f"{_name}: Average Rank")
            )
        )
        rank_variance_fig = go.Figure(
            layout=go.Layout(
                title=go.layout.Title(text=f"{_name}: Rank Variance")
            )
        )
        train_loss_fig = go.Figure(
            layout=go.Layout(
                title=go.layout.Title(text=f"{_name}: Train Loss")
            )
        )
        val_loss_fig = go.Figure(
            layout=go.Layout(
                title=go.layout.Title(text=f"{_name}: Validation Loss")
            )
        )

        # ---------------------------------------------------- 04
        # load history and ranks
        with open(self.history_file_path.as_posix(), 'rb') as file_pi:
            history = pickle.load(file_pi)
        # noinspection PyTypeChecker
        ranks = np.load(self.ranks_file_path)

        # ---------------------------------------------------- 05

    def wipe(self):
        if self.is_executing or not self.is_done:
            print(f" > {self.name} ... "
                  f"wiping files ...")
            for _ in self.store_dir.iterdir():
                _.unlink()
            self.store_dir.rmdir()

    @classmethod
    def get_existing_experiments_on_disk(cls) -> t.List["Experiment"]:
        _experiments_on_disk = []
        for _dataset_dir in ROOT_DIR.iterdir():
            for _model_dir in _dataset_dir.iterdir():
                for _es_no_es_dir in _model_dir.iterdir():
                    if _es_no_es_dir.name == "es":
                        _es_no_es = True
                    elif _es_no_es_dir.name == "no_es":
                        _es_no_es = False
                    else:
                        raise Exception(f"Unexpected values {_es_no_es_dir.name} for _es_no_es")
                    for _id_dir in _es_no_es_dir.iterdir():
                        _experiments_on_disk.append(
                            Experiment(
                                dataset=Dataset[_dataset_dir.name],
                                model=Model[_model_dir.name],
                                id=int(_id_dir.name),
                                early_stopping=_es_no_es,
                            )
                        )
        return _experiments_on_disk

    @classmethod
    def experiment_generator(cls) -> t.Iterable["Experiment"]:
        for _id in range(NUM_EXPERIMENTS):
            for _dataset in DEFAULT_PARAMS.keys():
                for _model in DEFAULT_PARAMS[_dataset].keys():
                    yield Experiment(
                        dataset=_dataset, model=_model, id=_id, early_stopping=False,
                    )
            for _dataset in DEFAULT_PARAMS.keys():
                for _model in DEFAULT_PARAMS[_dataset].keys():
                    yield Experiment(
                        dataset=_dataset, model=_model, id=_id, early_stopping=True,
                    )

    @classmethod
    def do_it(cls):
        for _experiment in cls.experiment_generator():
            # ------------------------------------------------ 01
            # if experiment is already complemented then skip
            if _experiment.is_done:
                print(f" > {_experiment.name} ... "
                      f"skipping as already completed ...")
                continue

            # ------------------------------------------------ 02
            # if experiment is executing the skip
            if _experiment.is_executing:
                print(f" > {_experiment.name} ... "
                      f"skipping as someone is executing it ...")
                continue

            # ------------------------------------------------ 03
            # create a semaphore for other threads to detect
            print(f" > {_experiment.name} ... will train and rank ...")
            _experiment.is_executing_file_path.touch()

            # ------------------------------------------------ 04
            # get params
            _params = _experiment.default_params

            # ------------------------------------------------ 05
            # get data
            tracesTrain, tracesVal, labelsTrain, labelsVal, X_attack_processed, targets, key_attack = \
                _params.preprocessor.apply(_experiment.dataset)

            # ------------------------------------------------ 06
            # get model
            _model_make_fn = _experiment.model.make_fn(_experiment.dataset)
            _model = _model_make_fn(
                input_size=tracesTrain.shape[1], learning_rate=_params.learning_rate
            )
            _model.summary()

            # ------------------------------------------------ 07
            # Ensure the data is in the right shape
            input_layer_shape = _model.get_layer(index=0).input_shape
            if len(input_layer_shape) == 2:
                tracesTrain_shaped = tracesTrain
                tracesVal_shaped = tracesVal
            elif len(input_layer_shape) == 3:
                tracesTrain_shaped = tracesTrain.reshape((tracesTrain.shape[0], tracesTrain.shape[1], 1))
                tracesVal_shaped = tracesVal.reshape((tracesVal.shape[0], tracesVal.shape[1], 1))
            else:
                raise Exception(f"Unknown shape {len(input_layer_shape)}")

            # ------------------------------------------------ 08
            # train the model
            print(f" > {_experiment.name} ... training ...")
            if _experiment.early_stopping:
                checkpoint = ModelCheckpoint(
                    _experiment.model_file_path.as_posix(),
                    monitor='val_loss', mode='min', verbose=1, save_best_only=True)
            else:
                checkpoint = ModelCheckpoint(
                    _experiment.model_file_path.as_posix(), verbose=1, save_best_only=False)

            if _params.one_cycle_lr:
                print('During training we will make use of the One Cycle learning rate policy.')
                lr_manager = OneCycleLR(max_lr=_params.learning_rate, end_percentage=0.2, scale_percentage=0.1,
                                        maximum_momentum=None, minimum_momentum=None, verbose=False)
                callbacks = [checkpoint, lr_manager]
            else:
                callbacks = [checkpoint]
            history = _model.fit(
                x=tracesTrain_shaped, y=to_categorical(labelsTrain, num_classes=256),
                validation_data=(tracesVal_shaped, to_categorical(labelsVal, num_classes=256)),
                batch_size=_params.batch_size, verbose=1, epochs=_params.epochs, callbacks=callbacks)

            # ------------------------------------------------ 09
            # save history
            with open(_experiment.history_file_path, 'wb') as _file:
                pickle.dump(history.history, _file)

            # ------------------------------------------------ 10
            # delete model
            del _model
            del history
            K.clear_session()

            # ------------------------------------------------ 11
            # perform ranking
            # load model
            _model = load_model(_experiment.model_file_path.as_posix())
            # adjust shape
            input_layer_shape = _model.get_layer(index=0).input_shape
            if len(input_layer_shape) == 2:
                tracesAttack_shaped = X_attack_processed
            elif len(input_layer_shape) == 3:
                tracesAttack_shaped = X_attack_processed.reshape(
                    (X_attack_processed.shape[0], X_attack_processed.shape[1], 1))
            else:
                raise Exception(f"Unknown shape {len(input_layer_shape)}")
            # get predictions
            print(f" > {_experiment.name} ... "
                  f"get predictions ...")
            predictions = _model.predict(tracesAttack_shaped, verbose=1)

            print(f" > {_experiment.name} ... "
                  f"computing ranks ...")
            ranks = compute_ranks(
                predictions=predictions,
                all_guess_targets=targets,
                correct_key=key_attack,
                num_attacks=NUM_ATTACKS_PER_EXPERIMENT,
            )

            # Calculate the mean of the rank over the nattack attacks
            avg_rank = np.mean(ranks, axis=0)

            print(np.where(avg_rank <= 0.))

            # ------------------------------------------------ 12
            # save ranks
            np.save(_experiment.ranks_file_path.as_posix(), ranks)

            # ------------------------------------------------ 13
            # as things are over release semaphore
            _experiment.is_executing_file_path.unlink()

    @classmethod
    def report_it(cls):

        # --------------------------------------------------- 01
        # wipe any stale results
        # cls.wipe_it()

        # --------------------------------------------------- 02
        # loop over - no early stopping
        print("Generating report ...")
        for _dataset in DEFAULT_PARAMS.keys():
            # ----------------------------------------------- 02.01
            # report md file
            _report_md_file_path = ROOT_DIR.parent / f"report_{_dataset.name}.md"
            _report_md_lines = [
                f"# Dataset {_dataset.name}: Analysis of {NUM_EXPERIMENTS} experiments", ""
            ]
            # derive some ranges
            _avg_rank_y_min = 0.
            _avg_rank_y_max = 0.
            _rank_variance_y_min = 0.
            _rank_variance_y_max = 0.
            # precompute certain things that are global to models used by this dataset
            for _es in [False, True]:
                for _model in DEFAULT_PARAMS[_dataset].keys():
                    for _id in range(NUM_EXPERIMENTS):
                        _experiment = Experiment(
                            dataset=_dataset, model=_model,
                            early_stopping=_es, id=_id,
                        )
                        if not (_experiment.ranks_file_path.exists() and _experiment.history_file_path.exists()):
                            continue
                        _ranks = _experiment.ranks
                        _avg_rank = np.mean(_ranks, axis=0)
                        _rank_variance = np.var(_ranks, axis=0)
                        # update y ranges
                        # noinspection PyArgumentList
                        _avg_rank_y_max = max(_avg_rank_y_max, _avg_rank.max())
                        # noinspection PyArgumentList
                        _rank_variance_y_max = max(_rank_variance_y_max, _rank_variance.max())

            # ----------------------------------------------- 02.02
            for _es in [False, True]:

                if _es:
                    _report_md_lines += [
                        f"## Modified original to work with early stopping", ""
                    ]
                else:
                    _report_md_lines += [
                        f"## Original (i.e. without early stopping)", ""
                    ]

                # lines for table
                _table_header = "|"
                _table_sep = "|"
                _table_avg_rank = "|"
                _table_rank_variance = "|"
                _table_train_loss = "|"
                _table_val_loss = "|"

                # violin fig data
                _violin_fig_data = {
                    'experiment_id': [],
                    'model': [],
                    'min traces needed for average rank to be zero': [],
                }
                _violin_failure_percent = {}
                _violin_y_max = 0

                for _model in DEFAULT_PARAMS[_dataset].keys():
                    _total_experiments = 0
                    _failed_experiments = 0

                    # create figures
                    _fig_name = f"{_dataset.name}-{_model.name} {'(with early stopping)' if _es else ''}"
                    _avg_rank_fig = go.Figure(
                        layout=go.Layout(
                            title=go.layout.Title(
                                text=f"Average Rank: {_fig_name}")
                        )
                    )
                    _rank_variance_fig = go.Figure(
                        layout=go.Layout(
                            title=go.layout.Title(
                                text=f"Rank Variance: {_fig_name}")
                        )
                    )
                    _train_loss_fig = go.Figure(
                        layout=go.Layout(
                            title=go.layout.Title(text=f"Train Loss: {_fig_name}")
                        )
                    )
                    _val_loss_fig = go.Figure(
                        layout=go.Layout(
                            title=go.layout.Title(text=f"Validation Loss: {_fig_name}")
                        )
                    )

                    # get all experiments with results available
                    for _id in range(NUM_EXPERIMENTS):
                        # get experiment
                        _experiment = Experiment(
                            dataset=_dataset, model=_model,
                            early_stopping=_es, id=_id,
                        )

                        # skip if not available
                        if not (_experiment.ranks_file_path.exists() and _experiment.history_file_path.exists()):
                            continue

                        # extract some data
                        _rank_plot_until = _dataset.rank_plot_until
                        _ranks = _experiment.ranks
                        _train_loss, _val_loss = _experiment.losses
                        _avg_rank = np.mean(_ranks, axis=0)
                        _rank_variance = np.var(_ranks, axis=0)
                        _total_experiments += 1
                        _traces_with_rank_0 = np.where(_avg_rank <= 0.0)[0]
                        _violin_fig_data['experiment_id'].append(_id)
                        _violin_fig_data['model'].append(_model.name)
                        if len(_traces_with_rank_0) == 0:
                            _failed_experiments += 1
                            _violin_fig_data['min traces needed for average rank to be zero'].append(
                                np.nan
                            )
                        else:
                            _traces_with_rank_0_min = _traces_with_rank_0.min()
                            _violin_fig_data['min traces needed for average rank to be zero'].append(
                                _traces_with_rank_0_min
                            )
                            _violin_y_max = max(_violin_y_max, _traces_with_rank_0_min)

                        # add to figure
                        _avg_rank_fig.add_trace(
                            go.Scatter(
                                x=np.arange(_rank_plot_until),
                                y=_avg_rank[:_rank_plot_until],
                                mode='lines',
                                name=f"exp_{_id:03d}",
                                showlegend=False,
                            )
                        )
                        _rank_variance_fig.add_trace(
                            go.Scatter(
                                x=np.arange(_rank_plot_until),
                                y=_rank_variance[:_rank_plot_until],
                                mode='lines',
                                name=f"exp_{_id:03d}",
                                showlegend=False,
                            )
                        )
                        _train_loss_fig.add_trace(
                            go.Scatter(
                                x=np.arange(len(_train_loss)),
                                y=_train_loss,
                                mode='lines',
                                name=f"exp_{_id:03d}",
                                showlegend=False,
                            )
                        )
                        _val_loss_fig.add_trace(
                            go.Scatter(
                                x=np.arange(len(_val_loss)),
                                y=_val_loss,
                                mode='lines',
                                name=f"exp_{_id:03d}",
                                showlegend=False,
                            )
                        )

                    # update y range for some figures
                    _avg_rank_fig.update_layout(yaxis_range=[_avg_rank_y_min, _avg_rank_y_max])
                    _rank_variance_fig.update_layout(yaxis_range=[_rank_variance_y_min, _rank_variance_y_max])

                    # save figures
                    _plot_relative_path = f"plots/{_dataset.name}/{_model.name}/{'es' if _es else 'no_es'}"
                    _plot_dir = ROOT_DIR.parent / _plot_relative_path
                    if not _plot_dir.exists():
                        _plot_dir.mkdir(parents=True)
                    _avg_rank_fig.write_image((_plot_dir / f"average_rank.svg").as_posix())
                    _rank_variance_fig.write_image((_plot_dir / f"rank_variance.svg").as_posix())
                    _train_loss_fig.write_image((_plot_dir / f"train_loss.svg").as_posix())
                    _val_loss_fig.write_image((_plot_dir / f"val_loss.svg").as_posix())

                    # make tabular report
                    # lines for table
                    if _failed_experiments > 0:
                        _failure_percent = (_failed_experiments / _total_experiments) * 100.
                        _table_success_status = f"<span style='color:red'> " \
                                                f"**{_failure_percent:.2f} % FAILURES** " \
                                                f"</span>"
                    else:
                        _failure_percent = 0.
                        _table_success_status = f"<span style='color:green'> " \
                                                f"**ALL SUCCESSES** " \
                                                f"</span>"
                    _violin_failure_percent[_model.name] = _failure_percent
                    _table_header += f"{_model.name}<br>{_table_success_status}|"
                    _table_sep += "---|"
                    _table_avg_rank += f"![Average Rank]({_plot_relative_path}/average_rank.svg)|"
                    _table_rank_variance += f"![Rank Variance]({_plot_relative_path}/rank_variance.svg)|"
                    _table_train_loss += f"![Train Loss]({_plot_relative_path}/train_loss.svg)|"
                    _table_val_loss += f"![Validation Loss]({_plot_relative_path}/val_loss.svg)|"

                # violin figure
                _violin_df = pd.DataFrame(_violin_fig_data)
                _violin_fig = px.violin(
                    _violin_df,
                    y="min traces needed for average rank to be zero",
                    x="model",
                    color="model",
                    box=False,
                    points="all",
                    hover_data=_violin_df.columns,
                    title="Distribution of min traces needed for average rank to be zero",
                )
                for _model_name, _failure_percent in _violin_failure_percent.items():
                    if _failure_percent == 0.:
                        _text = f" All passed "
                        _bgcolor = 'lightgreen'
                        _bordercolor = 'green'
                    else:
                        _text = f" {_failure_percent:.2f} % failed "
                        _bgcolor = 'pink'
                        _bordercolor = 'red'
                    _violin_fig.add_annotation(
                        x=_model_name, y=_violin_y_max + 2,
                        text=_text,
                        # bgcolor=_bgcolor,
                        bordercolor=_bordercolor,
                        showarrow=True,
                    )
                _violin_relative_path = \
                    f"plots/{_dataset.name}/{'violin_es.svg' if _es else 'violin_no_es.svg'}"
                _violin_fig_path = ROOT_DIR.parent / _violin_relative_path
                _violin_fig.write_image(_violin_fig_path.as_posix())

                # make table
                _report_md_lines += [
                    f"![Distribution of min traces needed for average rank to be zero]"
                    f"({_violin_relative_path})",
                    "",
                    _table_header, _table_sep,
                    _table_avg_rank, _table_rank_variance,
                    _table_train_loss, _table_val_loss
                ]

            # ----------------------------------------------- 02.03
            _report_md_file_path.write_text(
                "\n".join(_report_md_lines)
            )

    @classmethod
    def wipe_it(cls):
        for _e in cls.get_existing_experiments_on_disk():
            _e.wipe()

    @classmethod
    def zip_it(cls):
        _zip_file_path = pathlib.Path("results.zip")
        if _zip_file_path.exists():
            _zip_file_path.unlink()
        _zip_file = zipfile.ZipFile("results.zip", 'w')
        for _experiment in cls.get_existing_experiments_on_disk():
            if _experiment.is_done:
                _zip_file.write(
                    _experiment.history_file_path, arcname=_experiment.history_file_path.as_posix()
                )
                _zip_file.write(
                    _experiment.ranks_file_path, arcname=_experiment.ranks_file_path.as_posix()
                )
        _zip_file.close()


def main():

    print("*******************************************************************************")
    print("*******************************", sys.argv)
    print("*******************************************************************************")

    _mode = sys.argv[1]
    if _mode == 'do_it':
        Experiment.do_it()
    elif _mode == 'report_it':
        Experiment.report_it()
    elif _mode == 'wipe_it':
        Experiment.wipe_it()
    elif _mode == 'zip_it':
        Experiment.zip_it()
    else:
        raise Exception(f"Unknown {_mode}")

    print()
    print()
    print()


if __name__ == '__main__':
    main()