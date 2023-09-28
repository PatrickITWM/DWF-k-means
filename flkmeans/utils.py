import math
from itertools import accumulate
from typing import Optional, List, Dict

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

plt.style.use('seaborn-dark')


def distribute_to_clients(X: np.array,
                          n_clients: int = 3,
                          mode: str = "random",
                          distribution: Optional[List[float]] = None,
                          seed: Optional[int] = 1024) -> List[np.array]:
    """
    Receives a dataset X and splits it into small subsets. Returns a list of smaller datasets, each entry in the list
    correspond to the data of a client.

    :param X: A numpy array, the overall datset.
    :param n_clients: The number of clients.
    :param mode: The mode, how to distribute the data to the clients. Currently implemented modes:
            **random** (shuffles X and then slices the data into subsets of correct sizes according to the distribution),
            **clustered** (Runs one iteration round of k-means on X, where k = number of clients. Each client then gets
            the datapoints assigend to 'their' labels number).
            **half** Distributes half the data like in "random" and the other half like "clustered".
    :param distribution: (Optional) Only important if mode = random. A list of how much data each client should get.
            Sum of the list must be equal 1 and each entry must be positive.
    :param distribution: (Optional) seed for distribition.
    :return: A list of local datasets as numpy array.
    """
    DISTRIBUTION_MODES = ["random", "clustered"]

    if distribution is None:
        distribution = [1 / n_clients] * n_clients
    if mode == "random":
        # sizes of each chunk
        sizes = [math.floor(distribution[i] * len(X)) + (1 if i < len(X) % n_clients else 0) for i in range(n_clients)]
        # Build cumulated values for better processing
        acc = [0] + list(accumulate(sizes))
        # Shuffle
        X_copy = X.copy()
        np.random.seed(seed)
        np.random.shuffle(X_copy)
        # Create slices
        return [X_copy[acc[i]:acc[i + 1]] for i in range(n_clients)]
    if mode == "clustered":
        k_means = KMeans(n_clusters=n_clients, max_iter=5, n_init=5, random_state=seed)
        k_means.fit(X)
        labels = k_means.predict(X).tolist()
        return [np.array([x for x, c in zip(X, labels) if c == clus]) for clus in set(labels)]
    if mode == "half":
        X1, X2 = train_test_split(X, test_size=0.5, random_state=42)
        data_iid = distribute_to_clients(X1, mode="random", n_clients=n_clients, seed=seed)
        data_non_iid = distribute_to_clients(X2, mode="clustered", n_clients=n_clients, seed=seed)
        return [np.append(a, b, axis=0) for a, b in zip(data_iid, data_non_iid)]
    else:
        raise NotImplementedError(f"Distribution mode {mode} not implemented")


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
