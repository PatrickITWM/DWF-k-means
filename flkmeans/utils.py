import math
import random
from collections import defaultdict
from typing import Optional, List, Dict

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from sklearn.utils import shuffle
from tqdm.auto import trange


# plt.style.use('seaborn-dark')


def distribute_to_clients(X: np.ndarray,
                          y: np.ndarray,
                          clients_data_sizes: list[int],
                          p: float,
                          with_replacement=False,
                          cluster_exclusive_per_client=False,
                          seed: int = 1024,
                          verbose:bool=False):
    """
    Distributes a dataset X into subsets for multiple clients based on
    Chung, Jichan, Kangwook Lee, and Kannan Ramchandran. "Federated unsupervised clustering with generative models." AAAI 2022 international workshop on trustable, verifiable and auditable federated learning. Vol. 4. 2022.

    The p value controls the data heterogeneity.

    Parameters:
    X : np.ndarray
        The data matrix containing the features with rows representing samples and columns representing
        features.

    y : np.ndarray
        The labels array, where each entry corresponds to the label of the corresponding sample in the
        dataset. Necessary for distribution across clients based on the p value.

    clients_data_sizes : list[int]
        A list specifying the number of data points to assign to each client.

    p : float
        Proportion of data points to be allocated from clusters designated for specific clients.
        p=0 corresponds to uniform distribution across clients. p=1 assigns each client data from a single cluster.
        More specifically, the number of data points from a randomly chosen cluster assigned to each client is given by
         floor(p * n_samples), The rest is filled with randomly chosen data points from all clusters.

    with_replacement : bool, default False
        Whether to allow replacement when selecting data items from the clusters or remaining samples.

    cluster_exclusive_per_client : bool, default False
        Whether each client should exclusively receive data from a unique cluster. If False, two clients may be assigned
        the same cluster.

    seed : int, default 1024
        The random seed used for reproducible data distribution.

    Returns:
    list[np.ndarray]
        A list of data subsets, where each entry corresponds to the data assigned to a specific client.

    Raises:
    ValueError
        If the total data points are insufficient for distribution when `with_replacement` is False or if
        the number of clusters is inadequate for distribution across clients when
        `cluster_exclusive_per_client` is True. Note that we only do basic checks on the data, the input can still lead
        to error due to incompatible input (to less data per cluster to distribute etc.).
    """
    # Set the seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    # Copy for safety
    X = X.copy()
    y = y.copy()
    # Calculate basic objects we need to distribute the data
    n_clients = len(clients_data_sizes)
    cluster_indexes = list(set(y.tolist()))
    n_clusters = len(cluster_indexes)
    sizes_selected_cluster = [math.floor(p * s) for s in clients_data_sizes]
    sizes_rest = [math.ceil((1 - p) * s) for s in clients_data_sizes]
    X_splitted_into_clusters = {cluster_idx: X[y == cluster_idx].copy() for cluster_idx in cluster_indexes}

    #
    clients_data = []

    # Check if distribution is possible with given inputs
    if not with_replacement and len(X) < sum(clients_data_sizes):
        raise ValueError("Not enough data points to distribute to clients.")
    if cluster_exclusive_per_client and n_clusters < n_clients:
        raise ValueError("Not enough clusters to distribute to clients.")

    # Choose a cluster for each client
    if cluster_exclusive_per_client:
        selected_cluster_per_client = random.sample(cluster_indexes, k=n_clients)
    else:
        selected_cluster_per_client = random.choices(cluster_indexes, k=n_clients)
    # Distribute data from the selected cluster to the client
    for client in (trange(n_clients) if verbose else range(n_clients)):
        selected_cluster = selected_cluster_per_client[client]
        size = sizes_selected_cluster[client]
        X_selected_cluster = X_splitted_into_clusters[selected_cluster]
        if X_selected_cluster.shape[0] < size:
            raise ValueError(f"Not enough data points in cluster {selected_cluster} to distribute to client {client}. Available data points: {X_selected_cluster.shape[0]}, size: {size}")
        index = np.random.choice(X_selected_cluster.shape[0], size=size, replace=with_replacement)
        clients_data.append(X_selected_cluster[index].copy())
        if not with_replacement:
            X_splitted_into_clusters[selected_cluster] = np.delete(X_selected_cluster, index, axis=0)
    # Collect the remaining data in a single array again
    X_remainder = np.concatenate(list(X_splitted_into_clusters.values()), axis=0)
    # Distribute the remaining data to the clients
    for client in (trange(n_clients) if verbose else range(n_clients)):
        size = sizes_rest[client]
        index = np.random.choice(X_remainder.shape[0], size=size, replace=with_replacement)
        X_remainder_client = X_remainder[index].copy()
        clients_data[client] = np.concatenate([clients_data[client], X_remainder_client], axis=0)
        if not with_replacement:
            np.delete(X_remainder, index, axis=0)
    # shuffle the client data
    for client in range(n_clients):
        clients_data[client] = shuffle(clients_data[client], random_state=seed + client)
    return clients_data


def create_plot(ax: Axes,
                X: np.array,
                labels: Optional[List[int]] = None,
                centroids: Optional[List[List[float]]] = None,
                centroids_history: Optional[List[np.array]] = None,
                local_centroids_history: Optional[Dict[int, List[np.array]]] = None,
                cmap_str: str = "Set3",
                size_data: int = 10,
                size_centroids: int = 20,
                name="Name"):
    """
    Creates a nice looking plot visualizing the result of a clustering.

    :param ax: The axes object to draw on.
    :param X: The data set. Assumption: 2 dimensional.
    :param labels: A list of integers representing the corresponding label for x for all x in X.
    :param centroids:  A list of points (centroids) of dimension 2.
    :param centroids_history: A list of centroid matrices, the centroids during training.
    :param local_centroids_history: The history of centroids of each local client.
    :param cmap_str: The color map for plotting.
    :param size_data: The size of the data points in the plot.
    :param size_centroids: The size of the centroid points in the plot.
    :return: A matplotlib figure
    """
    cmap = plt.get_cmap(cmap_str)
    ax.set_aspect(1)  # sets the height to width ratio to 1
    # Compute necessary values
    if labels is not None:
        cluster_color = [cmap(c) for c in labels]
    else:
        cluster_color = "cornflowerblue"  # X.shape[0] * [cmap(1)]
    # Plot
    ax.scatter(X[:, 0], X[:, 1], s=size_data, c=cluster_color)  # Plot the data points
    #
    if local_centroids_history is not None:
        for client in range(max(local_centroids_history)):
            for centroid in range(len(centroids)):
                for i in range(len(centroids_history) - 1):
                    x = [centroids_history[i][centroid][0], local_centroids_history[client][i][centroid][0]]
                    y = [centroids_history[i][centroid][1], local_centroids_history[client][i][centroid][1]]
                    ax.plot(x, y, color="orange")
    #
    if centroids_history is not None:
        for i in range(len(centroids)):
            x = [centroid[i][0] for centroid in centroids_history]
            y = [centroid[i][1] for centroid in centroids_history]
            ax.plot(x, y, color="blue")
    #
    if centroids is not None:
        centroids_x = [c[0] for c in centroids]
        centroids_y = [c[1] for c in centroids]
        ax.scatter(centroids_x, centroids_y, alpha=1.0, s=size_centroids, color="black")  # Plot the centroids
    ax.set_title(name)

def map_pred_to_true(y_true, y_pred):
    """
    Returns a map to match the labels to the best true label.
    This is necessary, since the labels of the clustering (eg. 0-9)
    doesn't have to match the true labels (eg. A-J) or the label of the clustering doesn't match
    the true label (the clustering label 3 on MNIST data could correspond to the number 5 eg.).
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"len y_true: {len(y_true)}, len y_pred: {len(y_pred)}")
    labels = set(y_pred)
    dict_counts = {l: defaultdict(int) for l in labels}
    for i, (label_true, label_pred) in enumerate(zip(y_true, y_pred)):
        dict_counts[label_pred][label_true] += 1
    dict_map = {}
    for label_pred in labels:
        dict_count_dict = dict_counts[label_pred]
        max_occurrence = max(dict_count_dict.values()) if len(dict_count_dict) > 0 else -1
        max_args = [key for key, value in dict_count_dict.items() if value == max_occurrence]
        label_assigend = max_args[0]
        dict_map[label_pred] = label_assigend
    return [dict_map[v] for v in y_pred]