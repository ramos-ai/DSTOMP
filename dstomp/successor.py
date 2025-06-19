from concurrent.futures import ThreadPoolExecutor
from os.path import join
from typing import List, Tuple

import matplotlib.pyplot as plt
import numba
import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from environment.gridworld import Actions, GridWorld, StateType

SuccessorMatrix = NDArray[np.floating]


@numba.jit(nopython=True)
def _update_successor_fast(
    successor_matrix: NDArray,
    current_state_idx: int,
    next_state_idx: int,
    one_hot_features: NDArray,
    successor_alpha: float,
    gamma: float,
) -> NDArray:
    successor_matrix[current_state_idx] = (1 - successor_alpha) * successor_matrix[
        current_state_idx
    ] + successor_alpha * (one_hot_features + gamma * successor_matrix[next_state_idx])
    return successor_matrix


@numba.jit(nopython=True)
def _compute_distances(features: NDArray, center: NDArray) -> NDArray:
    return np.sum((features - center) ** 2, axis=1)


class Successor:
    def __init__(
        self,
        env: GridWorld,
        successor_alpha: float = 0.1,
        gamma: float = 0.99,
        reward_awareness: bool = False,
    ):
        self.gamma = gamma
        self.reward_awareness = reward_awareness

        self.env = env
        self.unimportant_states_for_successor = [StateType.WALL]
        self.env.reset(
            states_to_exclude=self.unimportant_states_for_successor,
            set_as_property=True,
        )

        self.successor_alpha = successor_alpha
        self.successor: SuccessorMatrix = np.zeros(
            (self.env.num_states, self.env.num_states)
        )

        # Pre-compute one-hot state representations
        self.one_hot_states = np.zeros((self.env.num_states, self.env.num_states))
        for i in range(self.env.num_states):
            self.one_hot_states[i] = self.env.get_one_hot_state(
                self.env.state_idx_to_coordinates[i]
            )

        self.behavior_policy_probs: NDArray[np.floating] = (
            np.ones(self.env.num_actions) / self.env.num_actions
        )

    def save_successor(self, base_path: str):
        np.save(join(f"{base_path}", "successor_representation.npy"), self.successor)

    def get_subgoals(
        self,
        num_clusters: int | None = None,
        random_seed: int | None = 42,
        off_policy_steps: int = 50_000,
    ):
        self.get_successor(off_policy_steps)
        subagoals_state_idx, _, _ = self.cluster_successor(num_clusters, random_seed)
        subgoals_state = [
            self.env.state_idx_to_coordinates[subagoal_state_idx]
            for subagoal_state_idx in subagoals_state_idx
        ]
        return subagoals_state_idx, subgoals_state

    def get_successor(self, off_policy_steps: int = 50_000):
        self.env.reset(
            states_to_exclude=self.unimportant_states_for_successor,
            set_as_property=True,
        )

        # Pre-generate random actions for efficiency
        actions = np.random.choice(
            self.env.num_actions, size=off_policy_steps, p=self.behavior_policy_probs
        )

        current_state = self.env.current_state
        for action in tqdm(actions):
            action = Actions(action)
            next_state, reward, _ = self.env.step(action)

            current_state_idx = self.env.state_coordinates_to_idx[current_state]
            next_state_idx = self.env.state_coordinates_to_idx[next_state]

            # Use pre-computed one-hot states and JIT-compiled update
            one_hot_features = (
                self.one_hot_states[current_state_idx] * reward
                if self.reward_awareness
                else self.one_hot_states[current_state_idx]
            )
            self.successor = _update_successor_fast(
                self.successor,
                current_state_idx,
                next_state_idx,
                one_hot_features,
                self.successor_alpha,
                self.gamma,
            )

            current_state = next_state

        return self.successor

    def cluster_successor(
        self,
        num_clusters: int | None = None,
        random_seed: int | None = 42,
        max_iter: int = 300,
    ) -> Tuple[List[int], NDArray[np.int_], List[int]]:
        features = self.successor

        if num_clusters is None:
            num_clusters = self.get_num_clusters(
                test_cluster_range=(2, 10), random_seed=random_seed
            )

        kmeans = KMeans(
            n_clusters=num_clusters,
            init="k-means++",
            random_state=random_seed,
            max_iter=max_iter,
        )
        cluster_labels = kmeans.fit_predict(features)
        centers = kmeans.cluster_centers_

        # Find medoids and count sizes
        medoids: List[int] = []
        cluster_sizes = np.zeros(num_clusters, dtype=int)

        for cluster_idx in range(num_clusters):
            cluster_mask = cluster_labels == cluster_idx
            cluster_states = features[cluster_mask]
            cluster_sizes[cluster_idx] = len(cluster_states)

            distances = _compute_distances(cluster_states, centers[cluster_idx])
            medoid_idx = np.where(cluster_mask)[0][np.argmin(distances)]
            medoids.append(int(medoid_idx))

        self.__set_subgoals(medoids)

        return medoids, cluster_labels, cluster_sizes.tolist()

    def __parallel_cluster_evaluation(
        self, k: int, features: NDArray, random_seed: int
    ) -> Tuple[float, float]:
        kmeans = KMeans(
            n_clusters=k, init="k-means++", random_state=random_seed, max_iter=300
        )
        preds = kmeans.fit_predict(features)

        return (float(kmeans.inertia_), float(silhouette_score(features, preds)))

    def get_num_clusters(
        self,
        test_cluster_range: Tuple[int, int] = (2, 30),
        random_seed: int | None = 42,
        do_parallel_execution: bool = False,
    ) -> int:
        cluster_range = range(test_cluster_range[0], test_cluster_range[1])
        if do_parallel_execution:
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        self._parallel_cluster_evaluation,
                        k,
                        self.successor,
                        random_seed,
                    )
                    for k in cluster_range
                ]

                results = [future.result() for future in futures]
            inertias, silhouette_scores = zip(*results)
        else:
            inertias: List[float] = []
            silhouette_scores: List[float] = []

            for k in cluster_range:
                kmeans = KMeans(
                    n_clusters=k, init="k-means++", random_state=random_seed
                )
                preds = kmeans.fit_predict(self.successor)

                inertia = float(kmeans.inertia_)
                sil_score = float(silhouette_score(self.successor, preds))

                inertias.append(inertia)
                silhouette_scores.append(sil_score)

        self.__plot_inertia_and_silhouette(
            cluster_range, list(inertias), list(silhouette_scores)
        )
        return cluster_range[np.argmax(silhouette_scores)]

    def __set_subgoals(self, medoids: List[int]):
        for medoid in medoids:
            medoid_state = self.env.state_idx_to_coordinates[medoid]
            self.env.room_array[medoid_state[1], medoid_state[0]] = StateType.BOTTLENECK

    def __plot_inertia_and_silhouette(
        self,
        cluster_range: range,
        inertias: List[float],
        silhouette_scores: List[float],
    ):
        actual_range = range(cluster_range.start, cluster_range.start + len(inertias))

        fig, ax1 = plt.subplots()

        color = "tab:blue"
        ax1.set_xlabel("Number of clusters")
        ax1.set_ylabel("Inertia", color=color)
        ax1.plot(actual_range, inertias, color=color, marker="o", label="Inertia")
        ax1.tick_params(axis="y", labelcolor=color)

        ax2 = ax1.twinx()
        color = "tab:red"
        ax2.set_ylabel("Silhouette Score", color=color)
        ax2.plot(
            actual_range,
            silhouette_scores,
            color=color,
            marker="x",
            label="Silhouette Score",
        )
        ax2.tick_params(axis="y", labelcolor=color)

        fig.tight_layout()
        plt.title("Inertia and Silhouette Score vs Number of Clusters")
        plt.savefig(
            "inertia_silhouette_plot.png", transparent=True, bbox_inches="tight"
        )
        # plt.show()
        plt.close()
