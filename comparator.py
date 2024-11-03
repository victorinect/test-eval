"""
this file has not been runned and is first version from chatgpt, first the cluster-agent needs to be correct
"""
import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean, mahalanobis
from sklearn.preprocessing import StandardScaler

class StyleComparator:
    def __init__(self, reference_csv_path, dataset_csv_path, clusters, thresholds):
        """
        Initialize the StyleComparator with file paths and configuration.

        Parameters:
        - reference_csv_path: Path to the CSV file containing reference image attributes.
        - dataset_csv_path: Path to the CSV file containing dataset image attributes.
        - clusters: Dictionary mapping cluster names to lists of attribute names.
        - thresholds: Dictionary mapping cluster names to similarity thresholds.
        """
        self.reference_attributes = pd.read_csv(reference_csv_path)
        self.dataset_attributes = pd.read_csv(dataset_csv_path)
        self.clusters = clusters
        self.thresholds = thresholds

    def calculate_similarity(self, ref_values, ds_values, method="euclidean"):
        """
        Calculate similarity between reference and dataset values for a cluster.

        Parameters:
        - ref_values: Values for the reference image in a specific cluster.
        - ds_values: Values for a dataset image in the same cluster.
        - method: The similarity metric ('euclidean', 'mahalanobis', 'correlation').

        Returns:
        - similarity: Calculated similarity score.
        """
        if method == "euclidean":
            return 1 / (1 + euclidean(ref_values, ds_values))  # Inverse for similarity
        elif method == "mahalanobis":
            cov_matrix = np.cov(ds_values.T)  # Covariance matrix of dataset for Mahalanobis
            inv_cov_matrix = np.linalg.inv(cov_matrix) if np.linalg.det(cov_matrix) != 0 else np.eye(len(cov_matrix))
            return 1 / (1 + mahalanobis(ref_values, ds_values, inv_cov_matrix))
        elif method == "correlation":
            return pearsonr(ref_values, ds_values)[0]
        else:
            raise ValueError("Unsupported similarity method. Choose 'euclidean', 'mahalanobis', or 'correlation'.")


    def compare_styles(self):
        """
        Compare the style of the reference image to the dataset images.

        Returns:
        - result: Boolean indicating if the reference image passes the style check.
        """
        passes = 0
        total_clusters = len(self.clusters)

        for cluster, attributes in self.clusters.items():
            # Get reference and dataset attribute values for the current cluster
            ref_values = self.reference_attributes[attributes].values.flatten()
            similarity_scores = []

            for _, ds_row in self.dataset_attributes.iterrows():
                ds_values = ds_row[attributes].values.flatten()
                similarity = self.calculate_similarity(ref_values, ds_values)
                similarity_scores.append(similarity)

            # Check if the average similarity score exceeds the threshold
            avg_similarity = np.mean(similarity_scores)
            if avg_similarity >= self.thresholds[cluster]:
                passes += 1

        # Determine if the reference image passes the style check
        return passes >= total_clusters / 2  # At least half of the clusters must pass

# Example Usage
if __name__ == "__main__":
    reference_csv_path = "./results/underwater/reference-img-analysis_results.csv"
    dataset_csv_path = "./results/underwater/style_analysis_results.csv"

    # Define clusters and thresholds
    clusters = {
        'Cluster_1': ['line_density', 'shape_complexity', 'colorfulness'],
        'Cluster_2': ['texture_roughness', 'brightness'],
        'Cluster_3': ['contrast', 'symmetry', 'warmth_ratio'],
        'Cluster_4': ['proportion_of_space', 'edge_density']
    }
    thresholds = {
        'Cluster_1': 0.8,
        'Cluster_2': 0.75,
        'Cluster_3': 0.7,
        'Cluster_4': 0.85
    }

    comparator = StyleComparator(reference_csv_path, dataset_csv_path, clusters, thresholds)

    result = comparator.compare_styles()
    if result:
        print("The reference image matches the style of the dataset: PASS")
    else:
        print("The reference image does not match the style of the dataset: FAIL")
