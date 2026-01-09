from src.data_loader import IrisDataLoader
from src.models import KMeansWrapper, AgglomerativeWrapper, HDBSCANWrapper
from src.evaluation import ClusterEvaluator


def main():
    # 1. Load and Preprocess
    loader = IrisDataLoader()
    loader.load()
    X_scaled = loader.preprocess()

    # 2. Define Strategies
    # We use a list to easily iterate through different algorithms
    strategies = [
        KMeansWrapper(n_clusters=3),
        AgglomerativeWrapper(n_clusters=3),
        HDBSCANWrapper(min_cluster_size=3)
    ]

    # 3. Instantiate Evaluator
    evaluator = ClusterEvaluator(X_scaled)

    # 4. Run Analysis Loop
    print("Starting Clustering Analysis...")
    for strategy in strategies:
        # Fit model
        labels = strategy.fit_predict(X_scaled)

        # Evaluate metrics
        evaluator.evaluate(labels, strategy.name)

        # Visualize
        evaluator.visualize(labels, strategy.name)


if __name__ == "__main__":
    main()