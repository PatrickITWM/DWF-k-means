import math
import random
import time
from collections import Counter
from typing import Optional, List, Dict
from collections import deque

import numpy as np
from sklearn.cluster import KMeans, kmeans_plusplus, k_means


class FLKMeans:
    INIT_METHODS = ("random", "client_random", "kfed")
    AGGREGATION_METHODS = ("equal", "weighted_avg")
    BACKENDS = ("numpy", "sklearn")
    VERBOSE_LEVELS = (0, 1, 2)

    def __init__(self,
                 n_clusters: int = 5,
                 max_iter_global: int = 300,
                 min_iter_global: int = 5,
                 iter_local: int = 5,
                 n_init: int = 10,
                 init_method: str = "kfed",  # "client_random", "random"
                 n_client_random: int = 5,
                 aggregate_method: str = "equal",  # "weighted_avg"
                 time_of_weights_computation: str = "start",  # "last"
                 num_client_per_round: Optional[int] = None,
                 lr: Optional[float] = None,
                 momentum: Optional[float] = None,
                 steps_without_improvements: Optional[int] = None,
                 tol_global: float = 10 ** (-6),
                 tol_local: Optional[float] = None,
                 save_history: bool = False,
                 backend: str = "numpy",  # "sklearn"
                 verbose: int = 0):
        """
        Similar to the KMeans class of scikit learn (https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html).

        :param n_clusters: The number of clusters (k) and centroids.
        :param max_iter_global: The maximal number of global iterations.
                It stops earlier if the movement is under the given threshold (under a given tolerance).
        :param min_iter_global: The minimal number of global iterations. This number of iterations is guaranteed, the
                convergence check is only done after this number of iterations.
        :param iter_local: The maximal number of local iterations. The local (client side) iteration stops early
                if the movement is under the given threshold (under a given tolerance).
        :param n_init: KMeans itself is a greedy algorithm, which sometimes get stuck in a local minimum, which isn't
                optimal. To circumvent this problem, one can run k-means multiple times with different (randomly chosen)
                 initial centroids and take the best result. n_init specifies this number of runs. This is set to 1 if
                 init_method_ is "kfed".
        :param init_method: The method for choosing the initial centroids. This has a crucial impact on the number of
                rounds needed till convergence. Implemented options:
                 **random** (chooses k random initial vectors with entries between -1 and 1) and
                 **client_random** (for each k, chooses a random client and there n_client_random data points.
                    The initial centroid is considered as the average of these chosen points).
                **kfed** (The result of a one shot clustering [kfed] is used as starting points).
        :param n_client_random: Only necessary if init_method is client_random. Number of randomly chosen data points
                from a client. The initial centroid is defined as the average of these points.
        :param aggregate_method: The method of aggregating the trained local centroids to a set of global centroids.
                Implemented options:
                **equal** (the naive approach: just average the first centroid of all clients,
                the second centroids of all clients, ...) and
                **weighted_avg** (averages the i_th centroids of the clients with weights, which are changing each global
                learning round. The weight of centroid i of client c is given by the number of locally assigned data
                points (of client c) to the i_th centroid (on client c) after local training.
        :param time_of_weights_computation: On which centroids the weights are computed: With the global centroids
                (**start**) or with the last local centroids (**last**).
        :param num_client_per_round: Number of clients used per round. Must be smaller or equal to the number
                of all available clients and greater or equal to 1. The clients are chosen uniformly random each
                global learning round.
        :param lr: The learning rate. Necessary, if there is a lot of noise in the movement of the centroids.
                Choosing random subsets of clients each round can introduce a lot of noise. To reduce the noise, choose
                a sufficiently small value for the learning rate. However, this reduces convergence speed.
        :param momentum: To reduce the effect of the learning rate, one can specify a momentum.
        :param steps_without_improvements: Alternative way to stop global training early (that means, we consider the
                training as converged). If not None, this is used as additional stopping criterion together with the
                tolerance.
        :param tol_global: The tolerance, which specifies when the algorithm has converged (globally). If the
                movement (=||*||_2 norm of the matrix of old centroids minus the matrix of new centroids) is under the
                given tollerance, the algorithm stops. If too low, it will not converge because of too much noise,
                if not all clients are used each round.
        :param tol_local: The tolerance, which specifies when the local training has converged (client side).
                If not specified, use global tolerance.
        :param save_history: Weather the global centroids should be stored. Default is False.
        :param backend: The backend of computing the labels. 'numpy' for plain Python and numpy implementation or
                'sklearn' for using scikit learn on the clients.
        :param verbose: An integer specifiying how much information the algorithm will display during training. The
                higher, the more information is displayed.

        """
        # Check inputs
        if init_method not in FLKMeans.INIT_METHODS:
            raise ValueError(f"Expected one of {FLKMeans.INIT_METHODS} as init_method. Got {init_method}.")
        if aggregate_method not in FLKMeans.AGGREGATION_METHODS:
            raise ValueError(
                f"Expected one of {FLKMeans.AGGREGATION_METHODS} as init_method. Got {aggregate_method}.")
        if lr is not None and lr <= 0:
            raise ValueError(f"Learning rate has to be > 0. Got lr={lr}.")
        elif lr is not None and lr > 1:
            raise ValueError(f"Learning rate has to be <= 1. Got lr={lr}.")
        if backend not in FLKMeans.BACKENDS:
            raise ValueError(f"Backend {backend} is not implemented.")
        #
        self.n_clusters = n_clusters
        self.max_iter_global = max_iter_global
        self.min_iter_global = min(max_iter_global, min_iter_global)
        self.iter_local = iter_local
        self.n_init = n_init
        self.init_method = init_method
        self.n_client_random = n_client_random
        self.aggregate_method = aggregate_method
        self.time_of_weights_computation = time_of_weights_computation
        self.num_client_per_round = num_client_per_round
        self.lr = lr
        self.momentum = momentum
        self.steps_without_improvements = steps_without_improvements
        self.tol_global = tol_global
        self.tol_local = tol_local if tol_local is not None else tol_global
        self.save_history = save_history
        self.backend = backend
        self.verbose = verbose
        #
        self.centroids: np.array = None
        self.centroids_history: List[np.array] = []
        self.n_clients = None  # Will be added in fit, information is not available yet

    def _get_init_centroids(self, X_locals: List[np.array]) -> np.array:
        """
        Returns the initial centroids as a numpy array, where each column is a centroid. The dimension is
        (dimension of data) x (number of clusters/centroids)

        :param X_locals: A list of numpy arrays. The numpy arrays are the dataset of the "clients".
        :return: Initial centroids as numpy array.
        """
        # Choose the dataset of the 0-client to get the dimension of the data
        dim = X_locals[0].shape[1]
        # Choose right init method
        if self.init_method == "random":
            return 2 * np.random.rand(self.n_clusters, dim) - 1
        elif self.init_method == "client_random":
            centroids = []
            for _ in range(self.n_clusters):
                # For each centroid (which nees to be created), first choose a random client
                client = random.randrange(self.n_clients)
                # Select the data of the client, copy it (to prevent inplace modification) and shuffle it.
                X_copy = X_locals[client].copy()
                np.random.shuffle(X_copy)
                # Choose the first n_client_random data points from the dataset
                # The centroid is the average of these data points.
                centroid = np.mean(X_copy[:min(self.n_client_random, len(X_copy))], axis=0)
                centroids.append(centroid)
            # Convert to numpy array
            return np.array(centroids)
        elif self.init_method == "kfed":
            centroids_clients = []
            for i, X_client in enumerate(X_locals):
                try:
                    k_means = KMeans(n_clusters=self.n_clusters, n_init=5)
                    k_means.fit(X_client)
                    centroids = k_means.cluster_centers_
                    centroids_clients.append(centroids)
                except Exception as e:
                    centroids_clients.append(X_client)
                    print(e)
                    # centroids_clients.append(X_client)
            merged_centroids = np.concatenate(centroids_clients)
            k_means = KMeans(n_clusters=self.n_clusters, n_init=5)
            k_means.fit(merged_centroids)
            centroids = k_means.cluster_centers_
            return centroids

    def _client_update(self,
                       X_local: np.array,
                       centroids: np.array,
                       i_client: int = 0,
                       i_global: int = 0,
                       t_start: float = 0,
                       movement: float = 0,
                       min_movement: float = math.inf) -> np.array:
        """
        Local training on the given client. Receives the current global centroids, updates them in classical
        KMeans fashion on the local data and returns the updated centroids for client i_client.

        :param X_local: The local dataset of the client.
        :param centroids: The current global centroids.
        :param i_client: The index number of the client.
        :param i_global: The global round number.
        :param t_start: The starting time, when the algorithm has been startet.
        :param movement: The last global movement.
        :param min_movement: The overall minimal global movement.
        :return: Updated centroids.
        """
        # Print current status to console
        if self.verbose == 2:
            print(
                f"...Global round {i_global + 1}, "
                f"last global movement: {movement:.6f}, "
                f"min global movement: {min_movement:.6f}, "
                f"time: {time.time() - t_start:.2f}s, "
                f"client {i_client + 1}" + 40 * " ",
                flush=True, end="")
        # Do local_round many local training rounds or until converged.
        for local_round in range(self.iter_local):
            # One step of KMeans on the local data
            new_centroids = self._update_centroids(X_local, centroids)
            # Check how much the new centroids moved
            movement = np.linalg.norm(centroids - new_centroids)
            centroids = new_centroids
            # Check if converged
            if movement < self.tol_local:
                break
        # Update console
        if self.verbose == 2:
            print("\r", end="")
        # return local trained centroids
        return centroids

    def _update_centroids(self, X_local: np.array, centroids: np.array) -> np.array:
        """
        Performe one step of KMeans on the local data. returns the updated centroids

        :param X_local: Dataset of given client.
        :param centroids: current centroids.
        :return: Updated centroids.
        """
        # Assign the data to the closest centroid
        labels = self._assign_label(X_local, centroids)
        # The handler is crucial. In case no data point is assigned to a given labels, the centroid is not
        # moved/updated. Else the new centroid is the average of all assigned data points.
        empty_handler = lambda l, clus: l if len(l) > 0 else [centroids[clus]]
        # compute new centroids
        centroids_new = np.array(
            [np.mean(empty_handler(X_local[labels == clus], clus), axis=0) for clus in range(self.n_clusters)]
        )
        return centroids_new

    def _assign_label(self, X_local: np.array, centroids: np.array) -> np.array:
        """
        Assignes each data point in X_local to the closest centroid. Outputs a list of integers, where the integer
        correspond to the centroids index.

        :param X_local: Dataset (local).
        :param centroids: Given centroids.
        :return: A flat numpy array (list like) of indexes of the corresponding centroid to each data point.
        """
        if self.backend == "sklearn":
            centroids = centroids.astype("float32")
            X_local = X_local.astype("float32")
            dummy_k_means = KMeans(n_clusters=self.n_clusters,
                                   n_init=1,
                                   init=centroids,
                                   verbose=1)
            dummy_k_means.cluster_centers_ = centroids
            dummy_k_means._n_threads = 1
            return dummy_k_means.predict(X_local)
        elif self.backend == "numpy":
            # Compute all distances from each data point to each centroid.
            distances = np.array(
                [np.linalg.norm(X_local - centroids[k], axis=1) for k in range(self.n_clusters)]
            ).transpose()
            # distances = np.linalg.norm(X_local - centroids[:, None, :], axis=2).transpose() # slower!
            # The label for each data point is the one with the smallest distance
            labels = np.argmin(distances, axis=1)
            return labels

    def _score(self, X_local: np.array, centroids: np.array) -> float:
        """
        Computes the 'loss' of the given solution (centroids). The score is the sum of all squared distances from
        each data point to its assigned centroid.

        :param X_local: (Local) dataset.
        :param centroids: Current centroids.
        :return: The sum of all squared distances from each data point to its assigned centroid.
        """
        # Assign labels
        labels = self._assign_label(X_local, centroids)
        score = 0
        for clus in range(self.n_clusters):
            cluster_X_local = X_local[labels == clus]
            distances = np.linalg.norm(cluster_X_local - centroids[clus], axis=1) ** 2
            score += np.sum(distances)
        return score

    def _weights(self, X_local: np.array, centroids: np.array) -> np.array:
        """
        Computes the necessary weights for each centroid for the given labels. Only necessary if aggregate_mode =
        'weighted_avg'.

        :param X_local: Local dataset.
        :param centroids: Current centroids.
        :return: A flat numpy array, containing the number of data points assigned to centroid i for
        i=1,...,(number of centroids/clusters)
        """
        labels = self._assign_label(X_local, centroids)
        counter = Counter(labels)
        return np.array([counter.get(i, 0) for i in range(self.n_clusters)])

    def _aggregate(self,
                   X_locals: List[np.array],
                   local_centroids: List[np.array],
                   centroids: np.array,
                   centroids_movement: np.array,
                   clients_in_round: List[int]) -> np.array:
        """
        Receives a list of local centroids and aggregates them into one set of new (global) centroids.
        :param X_locals: A list of all client datasets.
        :param local_centroids:  A list of new local centroids, trained on active clients this round.
        :param centroids: The current global centroids.
        :param centroids_movement: The last movement of the global centroids.
        :param clients_in_round: A list of integers, specifying the clients participating this round.
        :return: The new aggregated, global centroids.
        """
        if self.aggregate_method == "equal":
            new_centroids = np.mean(local_centroids, axis=0)
        elif self.aggregate_method == "weighted_avg":
            # Get the weights for each active client this round
            if self.time_of_weights_computation == "start":
                weights = [self._weights(X_locals[c], centroids) for c in clients_in_round]
            elif self.time_of_weights_computation == "last":
                weights = [self._weights(X_locals[c], local_centroids[i]) for i, c in enumerate(clients_in_round)]
            else:
                raise NotImplementedError(f"Argument {self.time_of_weights_computation=} is not implemented.")
            # For normalization, we need the sum of all weights for each centroid
            absolute = np.sum(weights, axis=0)
            # Replace 0 with 1 to avoid dividing by 0
            places_with_absolute_zeroes = np.isclose(absolute, 0)
            absolute[places_with_absolute_zeroes] = 1
            weights = [
                [e if not places_with_absolute_zeroes[i] else 1 / len(clients_in_round) for i, e in enumerate(weight)]
                for weight in weights]
            # The sum of the weights has to be equal 1, so scale the weights
            # Attention: weight and absolute is a flat numpy array of length = (number of centroids/clusters)
            scaled_weights = [(weight / absolute) for weight in weights]
            # The new centroids are weighted sum of the local centroids
            new_centroids = np.sum(
                [(local_centroids[i].transpose() * scaled_weight).transpose() for i, scaled_weight in
                 enumerate(scaled_weights)],
                axis=0)
        else:
            raise NotImplementedError(f"Aggregation method {self.aggregate_method} not implemented.")
        # Apply learning rate and momentum
        if self.lr is not None:
            new_centroids = self.lr * new_centroids + (1 - self.lr) * centroids
        if self.momentum is not None:
            new_centroids += self.momentum * centroids_movement
        return new_centroids

    def _frobenius_norm_change_centroids(self,
                                         centroids_old: np.array,
                                         centroids_new: np.array) -> float:
        """
        Computes the movement between two centroids arrays.

        :param centroids_old: Old centroids.
        :param centroids_new: New centroids.
        :return: Movement from old to new centroids. This is given by the Frobenius norm.
        """
        return np.linalg.norm(centroids_old - centroids_new)

    def score(self, X: np.array):
        """
        Computes the scaled score of a given dataset X with the trained centroids.
        :param X:  A datset.
        :return: The scaled score. Scaling factor is 1/len(X).
        """
        return self._score(X, self.centroids) / len(X)

    def predict(self, X: np.array) -> np.array:
        """
        Predict labels for a given dataset X based on the trained centroids.
        :param X: A dataset.
        :return: A flat numpy array of label indices.
        """
        return self._assign_label(X, self.centroids)

    def fit(self, X_locals: List[np.array]):
        """
        Method for train FLKMeans. It trains KMeans in a FL setting.

        :param X_locals: A List of local datasets. The entry with index i is the dataset of client i.
        """
        # Set initial variables
        n_samples = sum(len(X_local) for X_local in X_locals)
        self.n_clients = len(X_locals)
        optimal_centroids = None
        optimal_score = None
        min_movement_global = math.inf
        t_start_absolute = time.time()
        # Run the FLKMeans n_init_round times and keep the best centroids of these runs.
        for n_init_round in range(self.n_init):
            # Set initial variables for each try
            t_start = time.time()
            centroids_history = []
            #
            centroids = self._get_init_centroids(X_locals)
            if self.save_history:
                centroids_history.append(centroids)
            centroids_movement = np.zeros(centroids.shape)
            movement = 0
            queue_last_movements = deque(maxlen=self.steps_without_improvements)
            queue_last_movements.append(math.inf)
            # Now do the (FL) KMeans steps
            for i_global in range(self.max_iter_global):
                # Choose random clients participating this round, if number is specified, else use all
                if self.num_client_per_round is not None:
                    clients_in_round = random.sample(range(self.n_clients), k=self.num_client_per_round)
                else:
                    clients_in_round = range(self.n_clients)
                # Print status to console
                if self.verbose == 1:
                    m = min(queue_last_movements)
                    print(
                        f"...Global round {i_global + 1}, "
                        f"time: {time.time() - t_start:.2f}s, "
                        f"movement: {movement:.6f}, "
                        f"steps since last improvement: {[i for i, e in enumerate(queue_last_movements) if e <= m][0]}, "
                        f"min movement: {min_movement_global:.6f}" + 50 * " ",
                        flush=True, end="")
                # Compute new centroids on each client
                local_centroids = [self._client_update(X_locals[c],
                                                       centroids,
                                                       c,
                                                       i_global,
                                                       t_start,
                                                       movement,
                                                       min_movement_global) for c in clients_in_round]

                # Aggregate to one new global centroid matrix
                new_centroids = self._aggregate(X_locals,
                                                local_centroids,
                                                centroids,
                                                centroids_movement,
                                                clients_in_round)
                # Store
                if self.save_history:
                    centroids_history.append(new_centroids)
                # Check how much the solution moved
                centroids_movement = new_centroids - centroids
                movement = np.linalg.norm(centroids_movement)
                min_movement_global = min(movement, min_movement_global)
                if self.steps_without_improvements is not None and i_global >= self.min_iter_global:
                    queue_last_movements.appendleft(movement)
                # Update
                centroids = new_centroids
                # Update console
                if self.verbose == 1:
                    print("\r", end="")
                # Stop if converged (movement under threshold)
                min_steps_done = i_global >= self.min_iter_global
                movement_under_threshold = movement < self.tol_global
                movement_not_improved = self.steps_without_improvements is not None and \
                                        len(queue_last_movements) == self.steps_without_improvements and \
                                        min(queue_last_movements) == queue_last_movements[-1]
                if min_steps_done and (movement_under_threshold or movement_not_improved):
                    t_end = time.time()
                    t_diff_round = t_end - t_start
                    t_diff_absolute = t_end - t_start_absolute
                    estimated = self.n_init / (n_init_round + 1) * t_diff_absolute
                    # Print status to console
                    if self.verbose > 0:
                        print(
                            f"Attempt {n_init_round + 1} converged after {i_global + 1} rounds. "
                            f"Round: {t_diff_round:.2f}s, "
                            f"total: {t_diff_absolute:.2f}s, "
                            f"estimated: {estimated:.2f}s" + 30 * " ")
                    # break out of the global iterations loop, but not out of the number of tries loop.
                    break

            else:
                # We land here if the algorithm did not converge in the specified number of iterations.
                t_end = time.time()
                t_diff_round = t_end - t_start
                t_diff_absolute = t_end - t_start_absolute
                estimated = self.n_init / (n_init_round + 1) * t_diff_absolute
                # Print status to console
                if self.verbose > 0:
                    print(
                        f"Attempt {n_init_round + 1} not converged. "
                        f"Round: {t_diff_round:.2f}s, "
                        f"total: {t_diff_absolute:.2f}s, "
                        f"estimated: {estimated:.2f}s" + 40 * " ")
            # After training check if the computed solution is better than the previous solution.
            score = sum([self._score(X_local, centroids) for X_local in X_locals]) / n_samples
            if optimal_score is None or score < optimal_score:
                # Update the current optimal solution if improved
                if self.verbose > 0:
                    print(f"---New optimal score {score}")
                optimal_score = score
                optimal_centroids = centroids
                self.centroids_history = centroids_history
        # After all tries, set the centroids to the best found solution.
        self.centroids = optimal_centroids


class KFed:
    VERBOSE_LEVELS = (0, 1, 2)

    def __init__(self,
                 n_clusters: int = 5,
                 max_iter_global: int = 300,
                 max_iter_local: int = 300,
                 n_init: int = 10,
                 num_client_per_round: Optional[int] = None,
                 tol_global: float = 10 ** (-6),
                 tol_local: Optional[float] = None,
                 verbose: int = 0):
        """
        Similar to the KMeans class of scikit learn (https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html).

        :param n_clusters: The number of clusters (k) and centroids.
        :param max_iter_global: The maximal number of global iterations.
                It stops earlier if the movement is under the given threshold (under a given tolerance).
        :param max_iter_local: The maximal number of local iterations (per client).
                It stops earlier if the movement is under the given threshold (under a given tolerance).
        :param n_init: KMeans itself is a greedy algorithm, which sometimes get stuck in a local minimum, which isn't
                optimal. To circumvent this problem, one can run k-means multiple times with different (randomly chosen)
                 initial centroids and take the best result. n_init specifies this number of runs.
        :param num_client_per_round: Number of clients used per round. Must be smaller or equal to the number
                of all available clients and greater or equal to 1. The clients are chosen uniformly random each
                global learning round.
        :param tol_global: The tolerance, which specifies when the algorithm has converged (globally). If the
                movement (=||*||_2 norm of the matrix of old centroids minus the matrix of new centroids) is under the
                given tollerance, the algorithm stops. If too low, it will not converge because of too much noise,
                if not all clients are used each round.
        :param tol_local: The tolerance, which specifies when the local training has converged (client side).
                If not specified, use global tolerance.
        :param verbose: An integer specifiying how much information the algorithm will display during training. The
                higher, the more information is displayed.
        """

        # Check inputs
        #
        self.n_clusters = n_clusters
        self.max_iter_global = max_iter_global
        self.max_iter_local = max_iter_local
        self.n_init = n_init
        self.num_client_per_round = num_client_per_round
        self.tol_global = tol_global
        self.tol_local = tol_local if tol_local is not None else tol_global
        self.verbose = verbose
        #
        self.centroids_clients: List[np.array] = []
        self.merged_centroids: np.array = None  # Will be added in fit
        self.n_clients = None  # Will be added in fit, information is not available yet
        self.local_centroids_history: Dict[int, List[np.array]] = None  # Will be added in fit
        self.global_k_means = KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter_global, tol=tol_global)

    def fit(self, X_locals: List[np.array]):
        """
        Method for train KFed. It trains KMeans in a one shot FL setting.

        :param X_locals: A List of local datasets. The entry with index i is the dataset of client i.
        """
        # Set initial variables
        n_samples = sum(len(X_local) for X_local in X_locals)
        self.n_clients = len(X_locals)
        t_start_absolute = time.time()
        selected_clients = random.sample(range(self.n_clients), self.num_client_per_round)
        for i, X_client in enumerate(X_locals):
            if i not in selected_clients:
                continue
            if self.verbose >= 1:
                print(f"Run clustering on client {i}")
            try:
                k_means = KMeans(n_clusters=self.n_clusters,
                                 n_init=self.n_init,
                                 max_iter=self.max_iter_local,
                                 tol=self.tol_local)
                k_means.fit(X_client)
                centroids = k_means.cluster_centers_
                self.centroids_clients.append(centroids)
            except Exception as e:
                print(e)
        merged_centroids = np.concatenate(self.centroids_clients)
        self.merged_centroids = merged_centroids
        self.global_k_means.fit(merged_centroids)
        t_end_absolute = time.time()
        if self.verbose >= 1:
            print(f"Training abgeschlossen. Dauer: {t_end_absolute - t_start_absolute:.2f}s.")

    @property
    def centroids(self):
        return self.global_k_means.cluster_centers_

    def predict(self, X):
        return self.global_k_means.predict(X)


def score(X: np.array, centroids: np.array) -> float:
    """
        Computes the 'loss' of the given solution (centroids). The score is the sum of all squared distances from
        each data point to its assigned centroid.

        :param X:  dataset.
        :param centroids: Current centroids.
        :return: The sum of all squared distances from each data point to its assigned centroid.
        """
    n_clusters = centroids.shape[0]
    # Assign labels
    # Compute all distances from each data point to each centroid.
    distances = np.array(
        [np.linalg.norm(X - centroids[k], axis=1) for k in range(n_clusters)]
    ).transpose()
    # distances = np.linalg.norm(X_local - centroids[:, None, :], axis=2).transpose() # slower!
    # The label for each data point is the one with the smallest distance
    labels = np.argmin(distances, axis=1)
    # compute score
    score = 0
    for clus in range(n_clusters):
        cluster_X = X[labels == clus]
        distances = np.linalg.norm(cluster_X - centroids[clus], axis=1) ** 2
        score += np.sum(distances)
    return score / len(X)


class FKM:
    def __init__(self,
                 n_clusters=5,
                 max_iter_global: int = 300,
                 max_iter_local: int = 5,
                 num_client_per_round: int = 1,
                 tol_global: float = 10 ** (-6),
                 tol_local: Optional[float] = None,
                 verbose: int = 0,
                 seed: Optional[int] = None):
        self.n_clusters = n_clusters
        self.max_iter_global = max_iter_global
        self.max_iter_local = max_iter_local
        self.num_client_per_round = num_client_per_round
        self.tol_global = tol_global
        self.tol_local = tol_local if tol_local is not None else tol_global
        self.verbose = verbose if verbose is not None else 0
        self.seed = seed if seed is not None else 13
        self.centroids = None

    def _get_init_centroids(self, X_locals: List[np.array]):
        return [kmeans_plusplus(X, n_clusters=self.n_clusters, random_state=self.seed)[0] for X in X_locals]

    def _assign_label(self, X_local: np.array, centroids: np.array) -> np.array:
        """
        Assignes each data point in X_local to the closest centroid. Outputs a list of integers, where the integer
        correspond to the centroids index.

        :param X_local: Dataset (local).
        :param centroids: Given centroids.
        :return: A flat numpy array (list like) of indexes of the corresponding centroid to each data point.
        """
        centroids = centroids.astype("float32")
        X_local = X_local.astype("float32")
        dummy_k_means = KMeans(n_clusters=self.n_clusters,
                               n_init=1,
                               init=centroids,
                               verbose=1)
        dummy_k_means.cluster_centers_ = centroids
        dummy_k_means._n_threads = 1
        return dummy_k_means.predict(X_local)

    def _weights(self, X: np.array, centroids: np.array) -> List:
        """
        Computes the necessary weights for each centroid for the given labels. Only necessary if aggregate_mode =
        'weighted_avg'.X

        :param X: Local dataset.
        :param centroids: Current centroids.
        :return: A flat numpy array, containing the number of data points assigned to centroid i for
        i=1,...,(number of centroids/clusters)
        """
        labels = self._assign_label(X, centroids)
        counter = Counter(labels)
        return [counter.get(i, 0) for i in range(centroids.shape[0])]

    def fit(self, X_locals: List[np.array]):
        self.n_clients = len(X_locals)
        initialized = False
        # Start training
        for round in range(self.max_iter_global):
            if self.verbose >= 1:
                print(f'Global round {round}')
            # Select random clients
            clients_in_round = random.sample(range(self.n_clients), k=self.num_client_per_round)
            selected_X = [X_locals[c] for c in clients_in_round]
            # Create local centroids
            if not initialized:
                centroids_local = {c: centroid for c, centroid in
                                   zip(clients_in_round, self._get_init_centroids(selected_X))}
                weights = [self._weights(X, centroids) for centroids, X in zip(centroids_local.values(), selected_X)]
                initialized = True
            else:
                centroids_local = {}
                for c in range(self.num_client_per_round):
                    local_centroids = global_centroids.copy()
                    X_client = selected_X[c]
                    weights = self._weights(X_client, local_centroids)
                    filter_from_weights = [True if w > 0 else False for w in weights]
                    selected_local_centroids = local_centroids[filter_from_weights, :]
                    k = selected_local_centroids.shape[0]
                    try:
                        new_local_centroids = k_means(X_client,
                                                      n_clusters=k,
                                                      random_state=self.seed,
                                                      max_iter=self.max_iter_local,
                                                      tol=self.tol_local,
                                                      init=selected_local_centroids)[0]
                        centroids_local[c] = new_local_centroids
                    except ValueError as e:
                        if self.verbose >= 1:
                            print(e)
                weights = [self._weights(selected_X[c], centroids) for c, centroids in centroids_local.items()]
            # Collect local centroids and weights
            concatenated_centroids = np.concatenate(list(centroids_local.values()))
            concatenated_weights = np.concatenate(weights)
            # Do global k-means
            try:
                global_centroids = k_means(concatenated_centroids,
                                           n_clusters=self.n_clusters,
                                           random_state=self.seed,
                                           sample_weight=concatenated_weights,
                                           max_iter=self.max_iter_global,
                                           tol=self.tol_global)[0]
            except ValueError as e:
                if self.verbose >= 1:
                    print(e)
                    continue
            self.centroids = global_centroids

    def predict(self, X: np.array) -> np.array:
        """
        Predict labels for a given dataset X based on the trained centroids.
        :param X: A dataset.
        :return: A flat numpy array of label indices.
        """
        return self._assign_label(X, self.centroids)
