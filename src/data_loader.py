import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


class IrisDataLoader:
    """
    Responsible for loading, cleaning, and preprocessing the Iris dataset.
    """

    def __init__(self):
        self.data = None
        self.features = None
        self.target = None
        self.feature_names = None
        self.X_scaled = None

    def load(self):
        """Loads the Iris dataset from Scikit-Learn."""
        iris = load_iris()
        self.features = iris.data
        self.target = iris.target
        self.feature_names = iris.feature_names

        # Create a DataFrame for easier handling if needed later
        self.data = pd.DataFrame(self.features, columns=self.feature_names)
        print(f"Data Loaded: {self.data.shape[0]} samples with {self.data.shape[1]} features.")
        return self

    def preprocess(self):
        """
        Scales the features using StandardScaler.
        Crucial for distance-based algorithms like K-Means and HDBSCAN.
        """
        if self.features is None:
            raise ValueError("Data not loaded. Call load() first.")

        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(self.features)
        return self.X_scaled