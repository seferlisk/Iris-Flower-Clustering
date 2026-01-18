# Iris Flower Clustering Analysis

## Project Overview
This project performs an unsupervised clustering analysis on the famous Iris dataset to identify natural groupings of flowers based on their physical characteristics (Sepal Length, Sepal Width, Petal Length, Petal Width).

The solution is built using a modular **Object-Oriented Programming (OOP)** approach in Python, ensuring scalability and ease of maintenance.

## Directory Structure
```text
iris-clustering-project/
├── data/               # Data storage (if exporting to CSV)
├── src/                # Source code modules
│   ├── __init__.py
│   ├── data_loader.py  # Data loading and scaling logic
│   ├── models.py       # Wrapper classes for clustering algorithms
│   └── evaluation.py   # Metrics and PCA visualization
├── main.py             # Entry point script
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```
## Installation
Ensure you have Python 3.8+ installed.
Install the required dependencies:
```pip install -r requirements.txt```

## Usage
Run the main analysis script from the root directory:
```python main.py```
The script will:

1. Load and Standardize the Iris dataset.

2. Train three different clustering models (K-Means, Agglomerative, HDBSCAN).

3. Print evaluation metrics to the console.

4. Generate PCA-reduced 2D plots for each model.

## Methodologies
1. **Data Preprocessing**
     - Standardization: We use StandardScaler to scale features to unit variance. This is critical because distance-based algorithms like K-Means are sensitive to the magnitude of features (e.g., Petal Length vs. Sepal Width).
2. **Algorithms Used**
     - K-Means: A centroid-based algorithm that minimizes variance within clusters. It assumes clusters are spherical and of similar size.
     - Agglomerative Clustering: A hierarchical approach that merges data points from the bottom up. Useful for identifying nested structures.
     - HDBSCAN: A density-based algorithm that extends DBSCAN. It is robust to noise and can identify clusters of varying densities. It may assign outlier points a label of -1.
3. **Evaluation Metrics**
    We use intrinsic metrics since we are treating this as an unsupervised problem:
     - **Silhouette Score**: Measures how similar an object is to its own cluster compared to other clusters. Range: $[-1, 1]$. Higher is better.
     - **Davies-Bouldin Index**: Measures the average similarity ratio of each cluster with its most similar cluster. Lower is better.

## Visualization
Since the dataset has 4 dimensions, we use Principal Component Analysis (PCA) to reduce the data to 2 dimensions for visualization purposes. This allows us to plot the resulting clusters on a 2D scatter plot.