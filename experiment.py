# %%
# Imports

import os
import random
from typing import Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from keras.datasets import mnist
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score, v_measure_score, completeness_score, homogeneity_score
from tqdm import tqdm

from experiment_global_config import *
from flkmeans import FLKMeans, KFed, score
from flkmeans.utils import distribute_to_clients, create_plot
from utils import Saveable
from utils import map_pred_to_true


# %%
# Definitions

def random_points(n: int) -> list[list[float]]:
    """
    Function to get n random outliers. The dimension of the points is 2.
    The coordinates of the outliers are chosen equally from [-0.8, 1).
    """
    random.seed(1)
    return [[2 * random.random() - 1 for _ in range(2)] for _ in range(n)]


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
        self.path_save_points = "savepoints" + os.sep + f"{name}" + os.sep + f"{self.n_clients_per_round} of {self.n_clients}"
        self.savepoint_name = "{}, run {}, seed {}"  # name model, run number, seed (n_clients_per_round)

        VERBOSE_ALG = 0
        init_method = "kfed"

        create_folder_path_if_necessary(FOLDER_PATH / AUTHOR / EXPERIMENT)
        create_folder_path_if_necessary(self.path_save_points)

        self.k_means_full = [KMeans(n_clusters=k,
                                    n_init=1,
                                    max_iter=max_iter,
                                    random_state=i) for i in range(repetitions)]

        # flkmeans
        self.ewf_iid = [FLKMeans(n_clusters=k,
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
                                 verbose=VERBOSE_ALG,
                                 backend="sklearn",
                                 tol_global=tol_global,
                                 tol_local=tol_local) for _ in range(repetitions)]
        self.dwf_iid = [FLKMeans(n_clusters=k,
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
                                 verbose=VERBOSE_ALG,
                                 backend="sklearn",
                                 tol_global=tol_global,
                                 tol_local=tol_local) for _ in range(repetitions)]
        self.ewf_half_iid = [FLKMeans(n_clusters=k,
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
                                      verbose=VERBOSE_ALG,
                                      backend="sklearn",
                                      tol_global=tol_global,
                                      tol_local=tol_local) for _ in range(repetitions)]
        self.dwf_half_iid = [FLKMeans(n_clusters=k,
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
                                      verbose=VERBOSE_ALG,
                                      backend="sklearn",
                                      tol_global=tol_global,
                                      tol_local=tol_local) for _ in range(repetitions)]
        self.ewf_non_iid = [FLKMeans(n_clusters=k,
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
                                     verbose=VERBOSE_ALG,
                                     backend="sklearn",
                                     tol_global=tol_global,
                                     tol_local=tol_local) for _ in range(repetitions)]
        self.dwf_non_iid = [FLKMeans(n_clusters=k,
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
                                     verbose=VERBOSE_ALG,
                                     backend="sklearn",
                                     tol_global=tol_global,
                                     tol_local=tol_local) for _ in range(repetitions)]

        self.kfed_iid = [KFed(n_clusters=k,
                              max_iter_global=max_iter,
                              max_iter_local=max_iter,
                              n_init=1,
                              num_client_per_round=n_clients_kfed,
                              verbose=VERBOSE_ALG) for _ in range(repetitions)]
        self.kfed_half_iid = [KFed(n_clusters=k,
                                   max_iter_global=max_iter,
                                   max_iter_local=max_iter,
                                   n_init=1,
                                   num_client_per_round=n_clients_kfed,
                                   verbose=VERBOSE_ALG) for _ in range(repetitions)]
        self.kfed_non_iid = [KFed(n_clusters=k,
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
        return {"ewf_iid": self.ewf_iid,
                "ewf_half_iid": self.ewf_half_iid,
                "ewf_non_iid": self.ewf_non_iid,
                "dwf_iid": self.dwf_iid,
                "dwf_half_iid": self.dwf_half_iid,
                "dwf_non_iid": self.dwf_non_iid,
                "kfed_iid": self.kfed_iid,
                "kfed_half_iid": self.kfed_half_iid,
                "kfed_non_iid": self.kfed_non_iid}

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

    def train(self, X: np.array, seed: int = 1, parallel: bool = True):
        def train_model(model, X, i, seed, name):
            savepoint_name = self.savepoint_name.format(name, i, seed)
            try:
                model = Saveable.from_file(self.path_save_points + os.sep + savepoint_name + ".dill")
                return model
            except FileNotFoundError as e:
                if name.endswith("non_iid"):
                    data_clients_non_iid = distribute_to_clients(X, n_clients=N_CLIENTS, mode="clustered",
                                                                 seed=1993 * seed + 547 * i % 7919)
                    model.fit(data_clients_non_iid)
                elif name.endswith("half_iid"):
                    data_clients_half_iid = distribute_to_clients(X, n_clients=N_CLIENTS, mode="half",
                                                                  seed=1993 * seed + 547 * i % 7919)
                    model.fit(data_clients_half_iid)
                else:
                    data_clients_iid = distribute_to_clients(X, n_clients=N_CLIENTS, mode="random",
                                                             seed=1993 * seed + 547 * i % 7919)
                    model.fit(data_clients_iid)
                Saveable.save_object(model, self.path_save_points + os.sep + savepoint_name + ".dill")
                return model

        if self.verbose >= 1:
            print("Train k means full and partial, round")
        for i in tqdm(range(self.repetitions)):
            savepoint_name = f"k_means_full, run {i}, seed {seed}"
            try:
                Saveable.from_file(self.path_save_points + os.sep + savepoint_name + ".dill")
            except FileNotFoundError:
                self.k_means_full[i].fit(X)
                Saveable.save_object(self.k_means_full[i], self.path_save_points + os.sep + savepoint_name + ".dill")
                self.k_means_full[i] = None  # Remove from memory
        for name, model_list in self.models_fl.items():
            if self.verbose >= 1:
                print("----------------------")
                print("Train FL models, model ", name)
                print("----------------------")

            if parallel:
                # execute in parallel
                models = Parallel(n_jobs=N_KERNELS, verbose=20)(
                    delayed(train_model)(model, X, i, seed, name) for i, model in enumerate(model_list))
            else:
                models = []
                for i, model in enumerate(model_list):
                    print(f"Model number {i}")
                    train_model(model, X, i, seed, name)
        return self


# %%
# Create data

N_SAMPLES = 10_000
CENTERS = 5
CLUSTER_STD = 0.2
N_OUTLIERS = 100

# Synth data
X_synth, y_true = make_blobs(n_samples=N_SAMPLES,
                             centers=CENTERS,
                             cluster_std=CLUSTER_STD,
                             center_box=(-1, 1),
                             random_state=18042023)
# Add outliers
X_synth = np.append(X_synth, random_points(N_OUTLIERS), axis=0)
y_true = np.append(y_true, N_OUTLIERS * [CENTERS], axis=0)

# MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape((60_000, 28 * 28)).astype("float32") / 255.0
x_test = x_test.reshape((10_000, 28 * 28)).astype("float32") / 255.0

# Reduce data
x_train = x_train[:]
y_train = y_train[:]

# %%
# Plot synth data

fig, ax = plt.subplots(figsize=(5, 5))
create_plot(ax,
            X=X_synth,
            labels=y_true,
            name="Synthetic dataset",
            size_data=1,
            size_centroids=2,
            cmap_str="tab20")

# fig.savefig(FOLDER_PATH / AUTHOR / "Synthetic data" / "Synthetic dataset.png", dpi=300, format="png")

# %%

if "MNIST" in EXPERIMENT:
    X = x_train
    Y = y_train
elif "Synthetic data" in EXPERIMENT:
    X = X_synth
    Y = y_true

# %%

# Create experiments

experiments = {}
for n_clients_per_round in N_CLIENTS_PER_ROUND:
    experiment = Experiment(k=K,
                            n_clients=N_CLIENTS,
                            n_clients_per_round=n_clients_per_round,
                            n_clients_kfed=n_clients_per_round,
                            lr=LEARNING_RATE,
                            momentum=MOMENTUM,
                            tol_global=TOL_GLOBAL,
                            tol_local=TOL_LOCAL,
                            repetitions=REPETITIONS,
                            max_iter=MAX_ITER,
                            steps_without_improvements=STEPS_WITHOUT_IMPROVEMENTS,
                            name=EXPERIMENT,
                            verbose=1)
    experiments[n_clients_per_round] = experiment

# Here the code is run
for n, experiment in experiments.items():
    print(f"Train experiment with-----{n}/{N_CLIENTS}")
    experiment.train(X, n, parallel=PARALLEL)
    print("Save results")
    experiment.scores(X).to_csv(FOLDER_PATH / AUTHOR / EXPERIMENT / f"scores clients {n} out of {N_CLIENTS}.csv")
    experiment.accuracies(X, Y).to_csv(
        FOLDER_PATH / AUTHOR / EXPERIMENT / f"accuracies clients {n} out of {N_CLIENTS}.csv")
    experiment.v_measures(X, Y).to_csv(
        FOLDER_PATH / AUTHOR / EXPERIMENT / f"v measure clients {n} out of {N_CLIENTS}.csv")
    experiment.completeness_scores(X, Y).to_csv(
        FOLDER_PATH / AUTHOR / EXPERIMENT / f"completeness score clients {n} out of {N_CLIENTS}.csv")
    experiment.homogeneity_scores(X, Y).to_csv(
        FOLDER_PATH / AUTHOR / EXPERIMENT / f"homogeneity score clients {n} out of {N_CLIENTS}.csv")
