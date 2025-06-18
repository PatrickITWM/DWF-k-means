# %%
# Imports

import os
import random
import warnings
from itertools import product
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, Memory
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score, v_measure_score, completeness_score, homogeneity_score, \
    normalized_mutual_info_score, cohen_kappa_score
from tqdm import tqdm

from flkmeans import FLKMeans, KFed, score, FKM
from flkmeans.utils import distribute_to_clients, map_pred_to_true

warnings.filterwarnings('ignore')
os.environ["PYTHONWARNINGS"] = "ignore"

# %%
# Settings
# -------------------------------------
memory = Memory("savepoints", verbose=0)

# %%
# Settings
# -------------------------------------
RESULT_PATH = Path("experiments")
# -------------------------------------
N_KERNELS = -1
# -------------------------------------
EXPERIMENT = "SYNTHETIC"
K = 5
# -------------------------------------
N_CLIENTS = 100  # Ignored for FEMNIST
N_CLIENTS_PER_ROUND = range(10, 101, 10)
P = [0.0, 0.25, 0.5, 0.75, 1.0]
TOL_GLOBAL = 10 ** (-8)
TOL_LOCAL = 10 ** (-8)
LEARNING_RATE = 0.01
MOMENTUM = 0.8
STEPS_WITHOUT_IMPROVEMENTS = 300
REPETITIONS = 100
MAX_ITER = 10_000
MAX_ITER_FKM = 1_000


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
        except OSError as e:
            print(f"Could not create path {folder_path}")
            raise


# %%
# Create data

N_SAMPLES = 20_000
CENTERS = 5
CLUSTER_STD = 0.2
N_OUTLIERS = 200

clients_data_sizes = [100 - 50 + i for i in range(100)]

# Synth data
X, y = make_blobs(n_samples=N_SAMPLES,
                  centers=CENTERS,
                  cluster_std=CLUSTER_STD,
                  center_box=(-1, 1),
                  random_state=18042023)
# Add outliers
X = np.append(X, random_points(N_OUTLIERS), axis=0)
y = np.append(y, N_OUTLIERS * [CENTERS], axis=0)


# %%
# Heavy computation is cached
class MODELTYPE:
    KMEANS = "k-means"
    EWFKM = "EWF k-means"
    DWFKM = "DWF k-means"
    KFED = "k-FED"
    FKM = "FKM"

    @classmethod
    def all(cls):
        return [cls.KMEANS, cls.EWFKM, cls.DWFKM, cls.KFED, cls.FKM]


@memory.cache
def get_trained_k_means(experiment_name: str,
                        model_number: int,
                        p: float,
                        k: int,
                        n_clients: int,
                        n_clients_per_round: int,
                        lr: float,
                        momentum: float,
                        tol_global: float,
                        tol_local: float,
                        init_method: str,
                        max_iter: int = 10_000,
                        steps_without_improvements: Optional[int] = None):
    model = KMeans(n_clusters=k,
                   n_init=1,
                   max_iter=max_iter,
                   random_state=model_number)
    model.fit(X)
    return model


@memory.cache
def get_trained_ewf_k_means(experiment_name: str,
                            model_number: int,
                            p: float,
                            k: int,
                            n_clients: int,
                            n_clients_per_round: int,
                            lr: float,
                            momentum: float,
                            tol_global: float,
                            tol_local: float,
                            init_method: str,
                            max_iter: int = 10_000,
                            steps_without_improvements: Optional[int] = None):
    data = distribute_to_clients(X, y,
                                 clients_data_sizes=clients_data_sizes,
                                 p=p,
                                 with_replacement=False,
                                 clusters_to_clients_evenly_distributed=True,
                                 seed=547 * model_number % 7919)
    model = FLKMeans(n_clusters=k,
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
                     verbose=0,
                     backend="sklearn",
                     tol_global=tol_global,
                     tol_local=tol_local)
    model.fit(data)
    return model


@memory.cache
def get_trained_dwf_k_means(experiment_name: str,
                            model_number: int,
                            data_distribution: str,
                            k: int,
                            n_clients: int,
                            n_clients_per_round: int,
                            lr: float,
                            momentum: float,
                            tol_global: float,
                            tol_local: float,
                            init_method: str,
                            max_iter: int = 10_000,
                            steps_without_improvements: Optional[int] = None):
    data = distribute_to_clients(X, y,
                                 clients_data_sizes=clients_data_sizes,
                                 p=p,
                                 with_replacement=False,
                                 clusters_to_clients_evenly_distributed=True,
                                 seed=547 * model_number % 7919)
    model = FLKMeans(n_clusters=k,
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
                     verbose=0,
                     backend="sklearn",
                     tol_global=tol_global,
                     tol_local=tol_local)
    model.fit(data)
    return model


@memory.cache
def get_trained_k_fed(experiment_name: str,
                      model_number: int,
                      data_distribution: str,
                      k: int,
                      n_clients: int,
                      n_clients_per_round: int,
                      lr: float,
                      momentum: float,
                      tol_global: float,
                      tol_local: float,
                      init_method: str,
                      max_iter: int = 10_000,
                      steps_without_improvements: Optional[int] = None):
    data = distribute_to_clients(X, y,
                                 clients_data_sizes=clients_data_sizes,
                                 p=p,
                                 with_replacement=False,
                                 clusters_to_clients_evenly_distributed=True,
                                 seed=547 * model_number % 7919)
    model = KFed(n_clusters=k,
                 max_iter_global=max_iter,
                 max_iter_local=max_iter,
                 n_init=1,
                 num_client_per_round=n_clients_per_round,
                 verbose=0)
    model.fit(data)
    return model


@memory.cache
def get_trained_fkm(experiment_name: str,
                    model_number: int,
                    data_distribution: str,
                    k: int,
                    n_clients: int,
                    n_clients_per_round: int,
                    lr: float,
                    momentum: float,
                    tol_global: float,
                    tol_local: float,
                    init_method: str,
                    max_iter: int = 10_000,
                    steps_without_improvements: Optional[int] = None):
    data = distribute_to_clients(X, y,
                                 clients_data_sizes=clients_data_sizes,
                                 p=p,
                                 with_replacement=False,
                                 clusters_to_clients_evenly_distributed=True,
                                 seed=547 * model_number % 7919)
    model = FKM(n_clusters=k,
                max_iter_global=max_iter,
                max_iter_local=5,
                num_client_per_round=n_clients_per_round,
                verbose=0,
                tol_global=tol_global,
                tol_local=tol_local,
                seed=model_number)
    model.fit(data)
    return model


@memory.cache
def get_trained_model(experiment_name: str,
                      model_type: str,
                      model_number: int,
                      p: float,
                      k: int,
                      n_clients: int,
                      n_clients_per_round: int,
                      lr: float,
                      momentum: float,
                      tol_global: float,
                      tol_local: float,
                      init_method: str,
                      max_iter: int = 10_000,
                      max_iter_fkm: int = 1000,
                      steps_without_improvements: Optional[int] = None):
    print(f"{model_type=}, {model_number=}, {p=}, {n_clients_per_round=}, {experiment_name= }")
    if model_type == MODELTYPE.KMEANS:
        model = get_trained_k_means(experiment_name,
                                    model_number,
                                    p,
                                    k,
                                    n_clients,
                                    n_clients_per_round,
                                    lr,
                                    momentum,
                                    tol_global,
                                    tol_local,
                                    init_method,
                                    max_iter,
                                    steps_without_improvements)
    elif model_type == MODELTYPE.EWFKM:
        model = get_trained_ewf_k_means(experiment_name,
                                        model_number,
                                        p,
                                        k,
                                        n_clients,
                                        n_clients_per_round,
                                        lr,
                                        momentum,
                                        tol_global,
                                        tol_local,
                                        init_method,
                                        max_iter,
                                        steps_without_improvements)
    elif model_type == MODELTYPE.DWFKM:
        model = get_trained_dwf_k_means(experiment_name,
                                        model_number,
                                        p,
                                        k,
                                        n_clients,
                                        n_clients_per_round,
                                        lr,
                                        momentum,
                                        tol_global,
                                        tol_local,
                                        init_method,
                                        max_iter,
                                        steps_without_improvements)
    elif model_type == MODELTYPE.KFED:
        model = get_trained_k_fed(experiment_name,
                                  model_number,
                                  p,
                                  k,
                                  n_clients,
                                  n_clients_per_round,
                                  lr,
                                  momentum,
                                  tol_global,
                                  tol_local,
                                  init_method,
                                  max_iter,
                                  steps_without_improvements)
    elif model_type == MODELTYPE.FKM:
        model = get_trained_fkm(experiment_name,
                                model_number,
                                p,
                                k,
                                n_clients,
                                n_clients_per_round,
                                lr,
                                momentum,
                                tol_global,
                                tol_local,
                                init_method,
                                max_iter_fkm,
                                steps_without_improvements)

    else:
        raise ValueError(f"Unknown model type {model_type}")
    return model


@memory.cache
def compute_score(experiment_name: str,
                  model_type: str,
                  model_number: int,
                  p: float,
                  k: int,
                  n_clients: int,
                  n_clients_per_round: int,
                  lr: float,
                  momentum: float,
                  tol_global: float,
                  tol_local: float,
                  init_method: str,
                  max_iter: int = 10_000,
                  max_iter_fkm: int = 1000,
                  steps_without_improvements: Optional[int] = None):
    model = model_dict[(model_type, p, model_number, n_clients_per_round)]
    if model_type == MODELTYPE.KMEANS:
        centroids = model.cluster_centers_
    else:
        centroids = model.centroids
    return score(X, centroids)


@memory.cache
def compute_accuracy(experiment_name: str,
                     model_type: str,
                     model_number: int,
                     p: float,
                     k: int,
                     n_clients: int,
                     n_clients_per_round: int,
                     lr: float,
                     momentum: float,
                     tol_global: float,
                     tol_local: float,
                     init_method: str,
                     max_iter: int = 10_000,
                     max_iter_fkm: int = 1000,
                     steps_without_improvements: Optional[int] = None):
    model = model_dict[(model_type, p, model_number, n_clients_per_round)]
    return accuracy_score(y, map_pred_to_true(y, model.predict(X)))


@memory.cache
def compute_kappa(experiment_name: str,
                  model_type: str,
                  model_number: int,
                  p: float,
                  k: int,
                  n_clients: int,
                  n_clients_per_round: int,
                  lr: float,
                  momentum: float,
                  tol_global: float,
                  tol_local: float,
                  init_method: str,
                  max_iter: int = 10_000,
                  max_iter_fkm: int = 1000,
                  steps_without_improvements: Optional[int] = None):
    model = model_dict[(model_type, p, model_number, n_clients_per_round)]
    return cohen_kappa_score(y, map_pred_to_true(y, model.predict(X)))


@memory.cache
def compute_v_measure(experiment_name: str,
                      model_type: str,
                      model_number: int,
                      p: float,
                      k: int,
                      n_clients: int,
                      n_clients_per_round: int,
                      lr: float,
                      momentum: float,
                      tol_global: float,
                      tol_local: float,
                      init_method: str,
                      max_iter: int = 10_000,
                      max_iter_fkm: int = 1000,
                      steps_without_improvements: Optional[int] = None):
    model = model_dict[(model_type, p, model_number, n_clients_per_round)]
    return v_measure_score(y, model.predict(X))


@memory.cache
def compute_nmi(experiment_name: str,
                model_type: str,
                model_number: int,
                p: float,
                k: int,
                n_clients: int,
                n_clients_per_round: int,
                lr: float,
                momentum: float,
                tol_global: float,
                tol_local: float,
                init_method: str,
                max_iter: int = 10_000,
                max_iter_fkm: int = 1000,
                steps_without_improvements: Optional[int] = None):
    model = model_dict[(model_type, p, model_number, n_clients_per_round)]
    return normalized_mutual_info_score(y, model.predict(X))


@memory.cache
def compute_completeness(experiment_name: str,
                         model_type: str,
                         model_number: int,
                         p: float,
                         k: int,
                         n_clients: int,
                         n_clients_per_round: int,
                         lr: float,
                         momentum: float,
                         tol_global: float,
                         tol_local: float,
                         init_method: str,
                         max_iter: int = 10_000,
                         max_iter_fkm: int = 1000,
                         steps_without_improvements: Optional[int] = None):
    model = model_dict[(model_type, p, model_number, n_clients_per_round)]
    return completeness_score(y, model.predict(X))


@memory.cache
def compute_homogeneity(experiment_name: str,
                        model_type: str,
                        model_number: int,
                        p: float,
                        k: int,
                        n_clients: int,
                        n_clients_per_round: int,
                        lr: float,
                        momentum: float,
                        tol_global: float,
                        tol_local: float,
                        init_method: str,
                        max_iter: int = 10_000,
                        max_iter_fkm: int = 1000,
                        steps_without_improvements: Optional[int] = None):
    model = model_dict[(model_type, p, model_number, n_clients_per_round)]
    return homogeneity_score(y, model.predict(X))


args = {"experiment_name": EXPERIMENT,
        "k": K,
        "n_clients": N_CLIENTS,
        "lr": LEARNING_RATE,
        "momentum": MOMENTUM,
        "tol_global": TOL_GLOBAL,
        "tol_local": TOL_LOCAL,
        "init_method": "kfed",
        "max_iter": MAX_ITER,
        "max_iter_fkm": MAX_ITER_FKM,
        "steps_without_improvements": STEPS_WITHOUT_IMPROVEMENTS}

# Train models
trained_models = Parallel(n_jobs=N_KERNELS, batch_size=1, verbose=20)(
    delayed(get_trained_model)(model_type=model_type,
                               p=p,
                               model_number=model_number,
                               n_clients_per_round=n_clients_per_round,
                               **args)
    for
    model_type, p, model_number, n_clients_per_round
    in
    product(MODELTYPE.all(),
            P,
            range(REPETITIONS),
            N_CLIENTS_PER_ROUND))

# Convert the result to a dictionary
model_dict = {(model_type, data_distribution, model_number, n_clients_per_round): trained_models[i] for
              i, (model_type, data_distribution, model_number, n_clients_per_round) in
              enumerate(product(MODELTYPE.all(),
                                P,
                                range(REPETITIONS),
                                N_CLIENTS_PER_ROUND)
                        )
              }

# Compute the scores and store them
model_type_list = []
p_list = []
model_number_list = []
n_clients_per_round_list = []
loss_list = []
accuracy_list = []
v_measure_list = []
homogeneity_list = []
completeness_list = []
nmi_list = []
kappa_list = []

total_combinations = len(list(product(MODELTYPE.all(),
                                      P,
                                      range(REPETITIONS),
                                      N_CLIENTS_PER_ROUND)))

for model_type, p, model_number, n_clients_per_round in tqdm(product(MODELTYPE.all(),
                                                                     P,
                                                                     range(REPETITIONS),
                                                                     N_CLIENTS_PER_ROUND),
                                                             total=total_combinations):
    loss = compute_score(model_type=model_type,
                         p=p,
                         model_number=model_number,
                         n_clients_per_round=n_clients_per_round,
                         **args)
    accuracy = compute_accuracy(model_type=model_type,
                                p=p,
                                model_number=model_number,
                                n_clients_per_round=n_clients_per_round,
                                **args)
    v_measure = compute_v_measure(model_type=model_type,
                                  p=p,
                                  model_number=model_number,
                                  n_clients_per_round=n_clients_per_round,
                                  **args)
    homogeneity = compute_homogeneity(model_type=model_type,
                                      p=p,
                                      model_number=model_number,
                                      n_clients_per_round=n_clients_per_round,
                                      **args)
    completeness = compute_completeness(model_type=model_type,
                                        p=p,
                                        model_number=model_number,
                                        n_clients_per_round=n_clients_per_round,
                                        **args)
    nmi = compute_nmi(model_type=model_type,
                      p=p,
                      model_number=model_number,
                      n_clients_per_round=n_clients_per_round,
                      **args)
    kappa = compute_kappa(model_type=model_type,
                          p=p,
                          model_number=model_number,
                          n_clients_per_round=n_clients_per_round,
                          **args)
    model_type_list.append(model_type)
    p_list.append(p)
    model_number_list.append(model_number)
    n_clients_per_round_list.append(n_clients_per_round)
    loss_list.append(loss)
    accuracy_list.append(accuracy)
    v_measure_list.append(v_measure)
    homogeneity_list.append(homogeneity)
    completeness_list.append(completeness)
    nmi_list.append(nmi)
    kappa_list.append(kappa)

df = pd.DataFrame({"Model": model_type_list,
                   "p": p_list,
                   "Iteration": model_number_list,
                   "Clients per Round": n_clients_per_round_list,
                   "Loss": loss_list,
                   "Accuracy": accuracy_list,
                   "V-Measure": v_measure_list,
                   "Homogeneity": homogeneity_list,
                   "Completeness": completeness_list,
                   "NMI": nmi_list,
                   "Kappa": kappa_list})

create_folder_path_if_necessary(RESULT_PATH / EXPERIMENT)
df.to_csv(RESULT_PATH / EXPERIMENT / f"Results.csv")
