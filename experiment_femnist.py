# %%
# Imports

import os
import random
import shutil
from collections import defaultdict
from tempfile import TemporaryDirectory
from typing import Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, v_measure_score, completeness_score, homogeneity_score
from tqdm import tqdm

from experiment_global_config import *
from flkmeans import FLKMeans, KFed, score
from utils import Saveable
from utils import map_pred_to_true

random.seed(4823)


# %%
# Definitions

def create_folder_path_if_necessary(folder_path):
    """
    Creates the specified folder if necessary.
    """
    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path)
        except OSError as e:  # Guard against race condition
            print(f"Could not create path {folder_path}")
            raise


def load_data(path_zip: str | Path) -> dict[int, dict[str, pd.DataFrame]]:
    """
    Opens the zip archive into a temporary folder and loads the data into pandas dataframes.
    The return of this function has the form {client_id: {'data': pd.Dataframe(), 'label': pd.Dataframe()}, ...}
    """
    # Create temporary folder in which the data is unpacked
    print(f"Unpack data of file '{path_zip}'")
    tmp = TemporaryDirectory()
    shutil.unpack_archive(path_zip, tmp.name, "zip")
    # Navigate into the (unique) subfolder
    subfolder = os.listdir(tmp.name)[0]
    path_data = Path(tmp.name) / subfolder
    # Create the data dict
    print(f"Data successfully unpacked. Load data.")
    data = defaultdict(dict)
    for i, name in tqdm(enumerate(os.listdir(path_data)), total=len(os.listdir(path_data))):
        # we iterate over all files in the dir
        client_id = int(name.split("_")[0])  # The number at the beginning of the file
        type_of_content = "data" if "data" in name.lower() else "label"  # data or label
        # of the file name before the suffix
        data[client_id][type_of_content] = pd.read_csv(path_data / name, header=None)
    return data


def process_data(data: dict[int, dict[str, pd.DataFrame]]):
    x_clients = [data[client]["data"] for client in data]
    X = pd.concat(x_clients, axis=0)
    y_true = pd.concat([data[client]["label"] for client in data], axis=0)
    return x_clients, X, y_true.values.flatten().tolist()


class Experiment(Saveable):
    def __init__(self,
                 k: int,
                 n_clients: int,
                 n_clients_per_round: int,
                 n_clients_kfed: int,
                 lr: float,
                 momentum: float,
                 tol_global: float,
                 tol_local: float,
                 max_iter=10_000,
                 repetitions: int = 100,
                 steps_without_improvements: Optional[int] = None,
                 verbose: int = 0,
                 name: str = "Name Experiment"):
        self.k = k
        self.n_clients = n_clients
        self.n_clients_per_round = n_clients_per_round
        self.n_clients_kfed = n_clients_kfed
        self.lr = lr
        self.momentum = momentum
        self.tol_global = tol_global
        self.tol_local = tol_local
        self.max_iter = max_iter
        self.repetitions = repetitions
        self.steps_without_improvements = steps_without_improvements

        self.verbose = verbose
        self.name = name
        self.path_save_points = "savepoints" + os.sep \
                                + f"{name}" + os.sep \
                                + f"{self.n_clients_per_round} of {self.n_clients}"
        self.savepoint_name = "{}, run {}"  # name model, run number

        VERBOSE_ALG = 0
        init_method = "kfed"

        create_folder_path_if_necessary(self.path_save_points)

        self.k_means_full = [KMeans(n_clusters=k,
                                    n_init=1,
                                    max_iter=max_iter,
                                    random_state=i) for i in range(repetitions)]

        # flkmeans
        self.flkm_equal = [FLKMeans(n_clusters=k,
                                    max_iter_global=max_iter,
                                    min_iter_global=16,
                                    n_init=1,
                                    iter_local=5,
                                    init_method=init_method,
                                    n_client_random=5,
                                    aggregate_method="equal",
                                    num_client_per_round=n_clients_per_round,
                                    lr=lr,
                                    momentum=momentum,
                                    steps_without_improvements=steps_without_improvements,
                                    backend="sklearn",
                                    verbose=VERBOSE_ALG,
                                    tol_global=tol_global,
                                    tol_local=tol_local) for _ in range(repetitions)]
        self.flkm_weighted = [FLKMeans(n_clusters=k,
                                       max_iter_global=max_iter,
                                       min_iter_global=16,
                                       n_init=1,
                                       iter_local=5,
                                       init_method=init_method,
                                       n_client_random=5,
                                       aggregate_method="weighted_avg",
                                       num_client_per_round=n_clients_per_round,
                                       lr=lr,
                                       momentum=momentum,
                                       steps_without_improvements=steps_without_improvements,
                                       backend="sklearn",
                                       verbose=VERBOSE_ALG,
                                       tol_global=tol_global,
                                       tol_local=tol_local) for _ in range(repetitions)]

        self.kfed = [KFed(n_clusters=k,
                          max_iter_global=max_iter,
                          max_iter_local=max_iter,
                          n_init=1,
                          num_client_per_round=n_clients_kfed,
                          verbose=VERBOSE_ALG) for _ in range(repetitions)]

    def load_model(self, name: str, i: int):
        return Saveable.from_file(
            self.path_save_points + os.sep
            + self.savepoint_name.format(name,
                                         i,
                                         self.n_clients_per_round)
            + ".dill")

    @property
    def models_classical(self):
        return {"k_means_full": self.k_means_full}

    @property
    def models_fl(self):
        return {"flkm_equal": self.flkm_equal,
                "flkm_weighted": self.flkm_weighted,
                "kfed": self.kfed}

    @property
    def models(self):
        return self.models_classical | self.models_fl

    def scores(self, X):
        classical_scores = {"k_means_full": Parallel(n_jobs=N_KERNELS, verbose=1)(
            delayed(lambda i: score(X, self.load_model("k_means_full", i).cluster_centers_))(i) for i in
            range(self.repetitions))}
        fl_scores = {name: Parallel(n_jobs=N_KERNELS, verbose=1)(
            delayed(lambda i: score(X, self.load_model(name, i).centroids))(i) for i in range(self.repetitions)) for
            name in self.models_fl}
        df = pd.DataFrame(classical_scores | fl_scores)
        return df

    def accuracies(self,
                   X: np.array,
                   y_true):
        acc = {name: Parallel(n_jobs=N_KERNELS, verbose=1)(
            delayed(lambda i: accuracy_score(y_true,
                                             map_pred_to_true(y_true,
                                                              self.load_model(name, i).predict(X)
                                                              )
                                             )
                    )(i)
            for i in range(self.repetitions)
        )
            for name in self.models}
        df = pd.DataFrame(acc)
        return df

    def v_measures(self,
                   X: np.array,
                   y_true):

        vm = {name: Parallel(n_jobs=N_KERNELS, verbose=1)(
            delayed(lambda i: v_measure_score(y_true,
                                              self.load_model(name, i).predict(X)
                                              )
                    )(i) for i in range(self.repetitions)
        )
            for name in self.models}
        df = pd.DataFrame(vm)
        return df

    def completeness_scores(self,
                            X: np.array,
                            y_true):
        cs = {name: Parallel(n_jobs=N_KERNELS, verbose=1)(
            delayed(lambda i: completeness_score(y_true,
                                                 self.load_model(name, i).predict(X)
                                                 )
                    )(i) for i in range(self.repetitions)
        )
            for name in self.models}
        df = pd.DataFrame(cs)
        return df

    def homogeneity_scores(self,
                           X: np.array,
                           y_true):

        hs = {name: Parallel(n_jobs=N_KERNELS, verbose=1)(
            delayed(lambda i: homogeneity_score(y_true,
                                                self.load_model(name,
                                                                i).predict(X)
                                                )
                    )(i) for i in range(self.repetitions)
        )
            for name in self.models}
        df = pd.DataFrame(hs)
        return df

    def train(self, x_clients, X, parallel: bool = True):
        def train_model(model, x_clients, X, i, name):
            savepoint_name = self.savepoint_name.format(name, i)
            try:
                model = Saveable.from_file(self.path_save_points + os.sep + savepoint_name + ".dill")
                return model
            except FileNotFoundError as e:
                if name.endswith("full"):
                    model.fit(X)
                else:
                    model.fit(x_clients)
                Saveable.save_object(model, self.path_save_points + os.sep + savepoint_name + ".dill")
                return model

        if self.verbose >= 1:
            print("Train k means full, round")
        for i in tqdm(range(self.repetitions)):
            savepoint_name = self.savepoint_name.format("k_means_full", i)
            try:
                model = Saveable.from_file(self.path_save_points + os.sep + savepoint_name + ".dill")
            except FileNotFoundError:
                self.k_means_full[i].fit(X)
                Saveable.save_object(self.k_means_full[i], self.path_save_points + os.sep + savepoint_name + ".dill")
                self.k_means_full[i] = None
        for name, model_list in self.models_fl.items():
            if self.verbose >= 1:
                print("----------------------")
                print("Train FL models, model ", name)
                print("----------------------")

            if parallel:
                # execute in parallel
                Parallel(n_jobs=N_KERNELS, verbose=20)(
                    delayed(train_model)(model, x_clients, X, i, name) for i, model in enumerate(model_list))
            else:
                for i, model in enumerate(model_list):
                    print(f"Model number {i}")
                    train_model(model, x_clients, X, i, name)
        return self


# %%

# Override global config
EXPERIMENT = "FEMNIST"
DATA_DIST = "IID"

path_iid_data_zip = Path("data femnist/IID_NEW.zip")
path_half_iid_data_zip = Path("data femnist/HIID_NEW.zip")
path_non_iid_data_zip = Path("data femnist/NIID_NEW.zip")

if DATA_DIST == "IID":
    data = load_data(path_iid_data_zip)
elif DATA_DIST == "HALF_IID":
    data = load_data(path_half_iid_data_zip)
else:
    data = load_data(path_non_iid_data_zip)

x_clients, X, y_true = process_data(data)
print(f"Clients: {len(x_clients)}, Samples: {len(X)}")
print("Data per client:")
print(sorted([len(d) for d in x_clients]))
N_CLIENTS = len(x_clients)

# %%
# Plot fenmist data

create_folder_path_if_necessary("data femnist" + os.sep + DATA_DIST)

fig, axs = plt.subplots(3, 5, figsize=(10, 7))
for i, ax in enumerate(axs.reshape(-1)):
    image = np.array(data[15 + i]["data"].iloc[0]).reshape((28, 28))
    label = data[15 + i]["label"].iloc[0]
    ax.imshow(image, cmap="gray")
    ax.set_title(label[0])
    ax.set_xticks([])
    ax.set_yticks([])
fig.savefig(f"data femnist" + os.sep + DATA_DIST + os.sep + "example_data.png", dpi=300)

# %%

# Create experiments

experiments = {}
for n_clients_per_round in N_CLIENTS_PER_ROUND:
    experiment = Experiment(k=K,
                            n_clients=len(x_clients),
                            n_clients_per_round=n_clients_per_round,
                            n_clients_kfed=n_clients_per_round,
                            lr=LEARNING_RATE,
                            momentum=MOMENTUM,
                            tol_global=TOL_GLOBAL,
                            tol_local=TOL_LOCAL,
                            repetitions=REPETITIONS,
                            max_iter=MAX_ITER,
                            steps_without_improvements=STEPS_WITHOUT_IMPROVEMENTS,
                            name=EXPERIMENT + os.sep + DATA_DIST,
                            verbose=1)
    experiments[n_clients_per_round] = experiment

# %%
# Run training in parallel

# result = Parallel(n_jobs=N_KERNELS, verbose=20)(
#     delayed(experiment.train)(X, i) for i, experiment in enumerate(experiments.values()))
#
# # Convert result
# for i, n_clients_per_round in enumerate(N_CLIENTS_PER_ROUND):
#     experiments[n_clients_per_round] = result[i]

# %%
# Linear
create_folder_path_if_necessary(FOLDER_PATH / AUTHOR / EXPERIMENT / DATA_DIST)

for n, experiment in experiments.items():
    print(f"Train experiment with-----{n}/{N_CLIENTS}")
    experiment.train(x_clients, X, parallel=True)
    print("Save score results")
    experiment.scores(X).to_csv(
        FOLDER_PATH / AUTHOR / EXPERIMENT / DATA_DIST / f"scores clients {n} von {N_CLIENTS}.csv")
    print("Save accuracy results")
    experiment.accuracies(X, y_true).to_csv(
        FOLDER_PATH / AUTHOR / EXPERIMENT / DATA_DIST / f"accuracies clients {n} von {N_CLIENTS}.csv")
    print("Save v_meassure results")
    experiment.v_measures(X, y_true).to_csv(
        FOLDER_PATH / AUTHOR / EXPERIMENT / DATA_DIST / f"v measure clients {n} von {N_CLIENTS}.csv")
    print("Save completeness scores")
    experiment.completeness_scores(X, y_true).to_csv(
        FOLDER_PATH / AUTHOR / EXPERIMENT / DATA_DIST / f"completeness score clients {n} von {N_CLIENTS}.csv")
    print("Save homogeneity scores")
    experiment.homogeneity_scores(X, y_true).to_csv(
        FOLDER_PATH / AUTHOR / EXPERIMENT / DATA_DIST / f"homogeneity score clients {n} von {N_CLIENTS}.csv")
