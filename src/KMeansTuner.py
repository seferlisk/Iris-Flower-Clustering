import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


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