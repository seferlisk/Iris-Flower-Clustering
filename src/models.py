from abc import ABC, abstractmethod
from sklearn.cluster import KMeans, AgglomerativeClustering, HDBSCAN

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
        self.model = HDBSCAN(min_samples=min_samples, min_cluster_size=min_cluster_size)
        self._name = "HDBSCAN"

    def fit_predict(self, X):
        return self.model.fit_predict(X)

    @property
    def name(self):
        return self._name