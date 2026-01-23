from src.data_loader import IrisDataLoader
from src.models import KMeansWrapper, AgglomerativeWrapper, HDBSCANWrapper
from src.evaluation import ClusterEvaluator

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as shc


def run_tuning(X_scaled):
    """Runs the tuning for optimal k using K-Means and Silhouette scores."""
    print("Running K-Means tuning...")
    k_range = range(2, 11)
    scores = []

    for k in k_range:
        model = KMeansWrapper(n_clusters=k)
        labels = model.fit_predict(X_scaled)
        scores.append(silhouette_score(X_scaled, labels))

    print(f"Optimal k based on Silhouette Score: {k_range[scores.index(max(scores))]}")
    print("(Note: Based on domain knowledge, we proceed with k=3 for K-Means and Agglomerative)")


def plot_dendrogram(X_scaled):
    """Plots the Dendrogram for Agglomerative Clustering."""
    plt.figure(figsize=(10, 6))
    plt.title("Iris Dataset Dendrogram (Ward Linkage)")
    shc.dendrogram(shc.linkage(X_scaled, method='ward'))
    plt.xlabel("Sample Index")
    plt.ylabel("Euclidean Distance")
    plt.axhline(y=10, color='r', linestyle='--')
    plt.show()


def main():
    print("--- 1. Loading and Preprocessing Data ---")
    loader = IrisDataLoader().load()
    X_scaled = loader.preprocess()

    print("\n--- 2. Parameter Tuning ---")
    run_tuning(X_scaled)
    plot_dendrogram(X_scaled)

    print("\n--- 3. Running Final Clustering Models ---")
    # Instantiate models with our chosen parameters
    kmeans = KMeansWrapper(n_clusters=3)
    agg = AgglomerativeWrapper(n_clusters=3)

    # HDBSCAN with copy=True to prevent warnings
    hdbscan = HDBSCANWrapper(min_cluster_size=5)
    hdbscan.model.set_params(copy=True)

    # Fit and get labels
    labels_kmeans = kmeans.fit_predict(X_scaled)
    labels_agg = agg.fit_predict(X_scaled)
    labels_hdbscan = hdbscan.fit_predict(X_scaled)

    print("\n--- 4. Generating Visual Comparison ---")
    # Extract original features for interpretable axes
    petal_length = loader.features[:, 2]
    petal_width = loader.features[:, 3]

    # Setup the plot grid
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plt.suptitle('Cluster Comparison: Petal Length vs. Petal Width', fontsize=16)

    # Plot 1: K-Means
    sns.scatterplot(x=petal_length, y=petal_width, hue=labels_kmeans,
                    palette='Set1', ax=axes[0], s=70, edgecolor='k')
    axes[0].set_title('K-Means (k=3)')
    axes[0].set_xlabel('Petal Length (cm)')
    axes[0].set_ylabel('Petal Width (cm)')

    # Plot 2: Agglomerative
    sns.scatterplot(x=petal_length, y=petal_width, hue=labels_agg,
                    palette='Set1', ax=axes[1], s=70, edgecolor='k')
    axes[1].set_title('Agglomerative (k=3)')
    axes[1].set_xlabel('Petal Length (cm)')

    # Plot 3: HDBSCAN
    sns.scatterplot(x=petal_length, y=petal_width, hue=labels_hdbscan,
                    palette='viridis', ax=axes[2], s=70, edgecolor='k')
    axes[2].set_title('HDBSCAN (min_cluster_size=5)')
    axes[2].set_xlabel('Petal Length (cm)')

    plt.tight_layout()
    plt.show()
    print("Analysis Complete.")


if __name__ == "__main__":
    main()