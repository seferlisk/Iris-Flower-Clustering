from abc import ABC, abstractmethod
from sklearn.cluster import KMeans, AgglomerativeClustering, HDBSCAN
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

class ClusteringStrategy(ABC):
    """
    Abstract Base Class (Interface) for clustering algorithms.
    """
    @abstractmethod
    def fit_predict(self, X):
        """Fits the model and returns cluster labels."""
        pass

    @property
    @abstractmethod
    def name(self):
        """Returns the name of the algorithm."""
        pass

class KMeansWrapper(ClusteringStrategy):
    def __init__(self, n_clusters=3, random_state=42):
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
        self._name = "K-Means"

    def fit_predict(self, X):
        return self.model.fit_predict(X)

    @property
    def name(self):
        return self._name

class AgglomerativeWrapper(ClusteringStrategy):
    def __init__(self, n_clusters=3):
        self.model = AgglomerativeClustering(n_clusters=n_clusters)
        self._name = "Agglomerative Clustering"

    def fit_predict(self, X):
        return self.model.fit_predict(X)

    @property
    def name(self):
        return self._name

class HDBSCANWrapper(ClusteringStrategy):
    def __init__(self, min_samples=5, min_cluster_size=5):
        # HDBSCAN is density-based; it finds cluster numbers automatically.
        # copy=True is added to comply with scikit-learn 1.10+ standards.
        self.model = HDBSCAN(min_samples=min_samples, min_cluster_size=min_cluster_size, copy=True)
        self._name = "HDBSCAN"

    def fit_predict(self, X):
        return self.model.fit_predict(X)

    @property
    def name(self):
        return self._name

class KMeansTuner:
    """
    Analyzes the dataset to find the optimal number of clusters for K-Means.
    """

    def __init__(self, X_scaled, max_k=10):
        self.X = X_scaled
        self.max_k = max_k
        self.inertias = []
        self.silhouette_scores = []
        self.k_range = range(1, self.max_k + 1)
        self.k_range_sil = range(2, self.max_k + 1)  # Silhouette requires at least 2 clusters

    def calculate_metrics(self):
        """Calculates Inertia and Silhouette scores for different values of k."""
        for k in self.k_range:
            model = KMeans(n_clusters=k, random_state=42, n_init='auto')
            labels = model.fit_predict(self.X)
            self.inertias.append(model.inertia_)

            # Silhouette score is only valid for k >= 2
            if k >= 2:
                sil_score = silhouette_score(self.X, labels)
                self.silhouette_scores.append(sil_score)

    def plot_elbow(self):
        """Plots the Elbow Method graph."""
        plt.figure(figsize=(8, 5))
        plt.plot(self.k_range, self.inertias, marker='o', linestyle='--', color='b')
        plt.title('Elbow Method for Optimal k')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia (WCSS)')
        plt.xticks(self.k_range)
        plt.grid(True)
        plt.show()

    def plot_silhouette(self):
        """Plots the Silhouette Scores graph."""
        plt.figure(figsize=(8, 5))
        plt.plot(self.k_range_sil, self.silhouette_scores, marker='o', linestyle='--', color='g')
        plt.title('Silhouette Scores for Optimal k')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.xticks(self.k_range_sil)
        plt.grid(True)
        plt.show()
