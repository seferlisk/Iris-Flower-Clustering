import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score


class ClusterEvaluator:
    """
    Handles the evaluation metrics and visualization of clusters.
    """

    def __init__(self, X_scaled):
        self.X = X_scaled

    def evaluate(self, labels, algo_name):
        """Prints metric scores to the console."""
        # HDBSCAN assigns -1 to noise.
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in labels else 0)

        print(f"\n--- Results for {algo_name} ---")
        print(f"Estimated number of clusters: {n_clusters}")

        # Only calculate metrics if valid clusters exist (>1 and not all noise)
        if n_clusters > 1:
            sil_score = silhouette_score(self.X, labels)
            db_score = davies_bouldin_score(self.X, labels)
            print(f"Silhouette Score: {sil_score:.3f} (Higher is better)")
            print(f"Davies-Bouldin Index: {db_score:.3f} (Lower is better)")
        else:
            print("Not enough clusters to calculate metrics.")

    def visualize(self, labels, algo_name):
        """
        Projects data to 2D using PCA and plots the clusters.
        """
        # PCA projection for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X)

        plt.figure(figsize=(8, 6))

        # Create Scatter Plot
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', edgecolor='k', s=70)

        plt.colorbar(scatter, label='Cluster Label')
        plt.title(f"{algo_name} Clustering (PCA Projection)")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.grid(True, alpha=0.3)
        plt.show()