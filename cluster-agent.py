import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, SpectralClustering
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import cosine_similarity


class ClusterAgent:
    """
    A class to perform clustering on dataset attributes based on their correlations and similarities.
    """
    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset
        self.scaled_data = None
        self.correlation_matrix = None
        self.affinity_matrix = None
        self.cluster_labels = None
        self.cluster_method = None
        self.num_clusters = None
        self.clustered_attributes = {}
        self.unclustered_attributes = []

        self.visualization_dir = './visualizations'
        os.makedirs(self.visualization_dir, exist_ok=True)

    def calculate_correlation_matrix(self, method: str = 'pearson') -> pd.DataFrame:
        """
        Calculates the correlation between each pair of attributes and saves a heatmap.

        Parameters:
            method (str): Type of correlation to compute ('pearson' or 'spearman').

        Returns:
            pd.DataFrame: The correlation matrix.
        """
        if method not in ['pearson', 'spearman']:
            raise ValueError("Unsupported correlation method. Choose 'pearson' or 'spearman'.")

        self.correlation_matrix = self.dataset.corr(method=method)

        # Replace any missing values with 0
        if self.correlation_matrix.isnull().values.any():
            print("Warning: Correlation matrix has missing values. Filling them with 0.")
            self.correlation_matrix.fillna(0, inplace=True)

        # Plot and save the heatmap
        self._save_heatmap(
            matrix=self.correlation_matrix,
            title=f'{method.capitalize()} Correlation Matrix Heatmap',
            filepath=os.path.join(self.visualization_dir, f'correlation_matrix_{method}.png'),
            cmap='coolwarm',
            annot=True,
            fmt=".2f"
        )

        print(f"Correlation matrix calculated and heatmap saved at {os.path.join(self.visualization_dir, f'correlation_matrix_{method}.png')}.")
        return self.correlation_matrix

    def standardize_data(self):
        """
        Scales the data so that each feature has a mean of 0 and a standard deviation of 1.
        This helps improve the performance of clustering algorithms.
        """
        scaler = StandardScaler()
        self.scaled_data = scaler.fit_transform(self.dataset)
        print("Data has been standardized.")

    def calculate_affinity_matrix(self, method: str = 'cosine') -> np.ndarray:
        """
        Calculates how similar each pair of attributes is and saves a heatmap.

        Parameters:
            method (str): Type of similarity to compute ('cosine').

        Returns:
            np.ndarray: The affinity (similarity) matrix.
        """
        if self.scaled_data is None:
            raise ValueError("Data not standardized. Please run standardize_data() first.")

        if method == 'cosine':
            self.affinity_matrix = cosine_similarity(self.scaled_data.T)
        else:
            raise ValueError("Unsupported affinity method. Only 'cosine' is supported.")

        # Replace any infinite or NaN values with 0
        if not np.all(np.isfinite(self.affinity_matrix)):
            print("Warning: Affinity matrix has non-finite values. Replacing them with 0.")
            self.affinity_matrix = np.nan_to_num(self.affinity_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        # Make sure all values are between 0 and 1
        self.affinity_matrix = np.clip(self.affinity_matrix, 0, 1)

        # Plot and save the affinity heatmap
        self._save_heatmap(
            matrix=self.affinity_matrix,
            title=f"Affinity Matrix Heatmap ({method.capitalize()} Similarity)",
            filepath=os.path.join(self.visualization_dir, f'affinity_matrix_heatmap_{method}.png'),
            cmap='viridis',
            annot=False,
            fmt=""
        )

        print(f"Affinity matrix calculated and heatmap saved at {os.path.join(self.visualization_dir, f'affinity_matrix_heatmap_{method}.png')}.")
        return self.affinity_matrix

    def calculate_clusters(
        self,
        method: str = 'spectral',
        num_clusters: int = 3,
        distance_threshold: float = 0.5,
        similarity_matrix_method: str = 'cosine'
    ) -> np.ndarray:
        """
        Groups the attributes into clusters using the chosen method and saves the results.

        Parameters:
            method (str): Clustering method to use ('kmeans', 'hierarchical', 'spectral').
            num_clusters (int): Number of clusters to form (for 'kmeans' and 'spectral').
            distance_threshold (float): Threshold for forming clusters (for 'hierarchical').
            similarity_matrix_method (str): Similarity measure to use ('cosine').

        Returns:
            np.ndarray: Labels indicating which cluster each attribute belongs to.
        """
        self.cluster_method = method
        self.num_clusters = num_clusters

        if self.correlation_matrix is None:
            raise ValueError("Correlation matrix not calculated. Please run calculate_correlation_matrix() first.")

        if method == 'kmeans':
            self.cluster_labels = KMeans(n_clusters=num_clusters, random_state=42).fit_predict(self.correlation_matrix)
        elif method == 'hierarchical':
            self.cluster_labels = self._hierarchical_clustering(distance_threshold)
        elif method == 'spectral':
            self.cluster_labels = self._spectral_clustering(similarity_matrix_method, num_clusters)
        else:
            raise ValueError("Unsupported clustering method. Choose 'kmeans', 'hierarchical', or 'spectral'.")

        # Process and save cluster information
        self._process_clusters(method)
        self._save_clustermap()

        print(f"Clustering done using {method} with {self.num_clusters} clusters. Results saved.")
        return self.cluster_labels

    def _hierarchical_clustering(self, distance_threshold: float) -> np.ndarray:
        """
        Performs hierarchical clustering based on a distance threshold.

        Parameters:
            distance_threshold (float): The distance to use when forming clusters.

        Returns:
            np.ndarray: Cluster labels for each attribute.
        """
        # Convert correlation to distance
        distance_matrix = 1 - self.correlation_matrix

        # Make sure the diagonal is zero
        np.fill_diagonal(distance_matrix.values, 0)

        # Replace any infinite or NaN values with 1
        if not np.all(np.isfinite(distance_matrix)):
            print("Warning: Distance matrix has non-finite values. Replacing them with 1.")
            distance_matrix.replace([np.inf, -np.inf], 1, inplace=True)
            distance_matrix.fillna(1, inplace=True)

        # Convert to a condensed distance matrix
        condensed_distance = squareform(distance_matrix.values, checks=False)

        # Perform hierarchical clustering
        linkage_matrix = linkage(condensed_distance, method='ward')

        # Assign clusters based on the distance threshold
        cluster_labels = fcluster(linkage_matrix, t=distance_threshold, criterion='distance')

        # Save the dendrogram plot
        dendrogram_path = os.path.join(self.visualization_dir, 'hierarchical_dendrogram.png')
        self._save_dendrogram(linkage_matrix, dendrogram_path)

        print(f"Hierarchical dendrogram saved at {dendrogram_path}.")
        return cluster_labels

    def _spectral_clustering(self, similarity_method: str, num_clusters: int) -> np.ndarray:
        """
        Performs spectral clustering based on the affinity matrix.

        Parameters:
            similarity_method (str): Method to calculate similarity ('cosine').
            num_clusters (int): Number of clusters to form.

        Returns:
            np.ndarray: Cluster labels for each attribute.
        """
        if self.affinity_matrix is None:
            self.calculate_affinity_matrix(method=similarity_method)

        spectral = SpectralClustering(
            n_clusters=num_clusters,
            affinity='precomputed',
            random_state=42,
            assign_labels='kmeans'
        )
        cluster_labels = spectral.fit_predict(self.affinity_matrix)
        return cluster_labels

    def _save_heatmap(
        self,
        matrix: pd.DataFrame,
        title: str,
        filepath: str,
        cmap: str,
        annot: bool,
        fmt: str
    ):
        """
        Creates and saves a heatmap for the given matrix.

        Parameters:
            matrix (pd.DataFrame): The data to plot.
            title (str): Title of the heatmap.
            filepath (str): Where to save the heatmap image.
            cmap (str): Color map to use.
            annot (bool): Whether to display the data values on the heatmap.
            fmt (str): Format of the annotations.
        """
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            matrix,
            annot=annot,
            fmt=fmt,
            cmap=cmap,
            square=True,
            cbar_kws={"shrink": 0.5},
            linewidths=0.5
        )
        plt.title(title)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300)
        plt.close()

    def _save_dendrogram(self, linkage_matrix: np.ndarray, filepath: str):
        """
        Creates and saves a dendrogram plot based on the linkage matrix.

        Parameters:
            linkage_matrix (np.ndarray): The linkage matrix from hierarchical clustering.
            filepath (str): Where to save the dendrogram image.
        """
        plt.figure(figsize=(12, 8))
        dendrogram(
            linkage_matrix,
            labels=self.correlation_matrix.index,
            leaf_rotation=90,
            leaf_font_size=10
        )
        plt.title("Hierarchical Clustering Dendrogram (Attributes)")
        plt.xlabel("Attributes")
        plt.ylabel("Distance")
        plt.tight_layout()
        plt.savefig(filepath, dpi=300)
        plt.close()

    def _process_clusters(self, method: str):
        """
        Organizes attributes into strong clusters or marks them as unclustered.

        Parameters:
            method (str): The clustering method used.
        """
        clusters_dict = {}
        correlation_threshold = 0.7  # Minimum average correlation to consider a strong cluster

        for cluster_label in np.unique(self.cluster_labels):
            # Adjust cluster numbering based on the method
            current_label = cluster_label if method == 'hierarchical' else cluster_label + 1
            attributes = self.correlation_matrix.index[self.cluster_labels == cluster_label].tolist()

            if len(attributes) < 2:
                self.unclustered_attributes.extend(attributes)
                continue

            # Calculate the average correlation within the cluster
            sub_corr = self.correlation_matrix.loc[attributes, attributes]
            upper_tri = sub_corr.where(np.triu(np.ones(sub_corr.shape), k=1).astype(bool))
            avg_corr = upper_tri.stack().mean() if not upper_tri.stack().empty else 0

            if avg_corr >= correlation_threshold:
                clusters_dict[current_label] = attributes
            else:
                self.unclustered_attributes.extend(attributes)

        self.clustered_attributes = clusters_dict

        print(f"Number of strong clusters formed: {len(self.clustered_attributes)}")
        if self.unclustered_attributes:
            print(f"Number of unclustered attributes: {len(self.unclustered_attributes)}")
            print(f"Unclustered Attributes: {self.unclustered_attributes}")

    def _save_clustermap(self):
        """
        Creates and saves a clustermap showing the correlations between attributes.
        """
        # Order attributes: clustered first, then unclustered
        ordered_attributes = list(
            {attr for cluster in self.clustered_attributes.values() for attr in cluster}
        )
        ordered_attributes += self.unclustered_attributes

        # Add any missing attributes
        missing_attrs = set(self.correlation_matrix.index) - set(ordered_attributes)
        ordered_attributes += list(missing_attrs)

        clustermap_path = os.path.join(self.visualization_dir, 'attributes_clustermap.png')
        sns.clustermap(
            self.correlation_matrix.loc[ordered_attributes, ordered_attributes],
            row_cluster=True,
            col_cluster=True,
            cmap='coolwarm',
            figsize=(12, 10),
            linewidths=0.5,
            annot=True,
            fmt=".2f"
        )
        plt.title("Clustermap of Strongly Correlated Attributes")
        plt.tight_layout()
        plt.savefig(clustermap_path, dpi=300)
        plt.close()

        print(f"Clustermap of attributes saved at {clustermap_path}.")

    def extract_clustered_attributes(self) -> dict:
        """
        Retrieves the clusters of attributes.

        Returns:
            dict: A dictionary where each key is a cluster number and the value is a list of attributes in that cluster.
        """
        if not self.clustered_attributes and not self.unclustered_attributes:
            raise ValueError("Clusters not calculated. Please run calculate_clusters() first.")

        print("Clustered attributes have been extracted.")
        return self.clustered_attributes

    def describe_clusters(self, clusters_dict: dict, top_n: int = 3):
        """
        Prints out the details of each cluster, listing the top N attributes.

        Parameters:
            clusters_dict (dict): The clusters to describe.
            top_n (int): Number of top attributes to show for each cluster.
        """
        print("\nCluster Descriptions:")
        for cluster, attributes in clusters_dict.items():
            print(f"\nCluster {cluster}:")
            print(f"Number of Attributes: {len(attributes)}")
            print("Attributes:")
            for attr in attributes[:top_n]:
                print(f" - {attr}")
            if len(attributes) > top_n:
                print(" ...")

        if self.unclustered_attributes:
            print("\nUnclustered Attributes:")
            for attr in self.unclustered_attributes:
                print(f" - {attr}")


def main():
    """
    Runs the clustering process: loads data, processes it, performs clustering, and displays results.
    """
    try:
        # Define the path to your dataset
        dataset_path = "./results/underwater/style_analysis_results.csv"

        # Check if the dataset exists
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"The dataset file was not found at the path: {dataset_path}")

        # Load the dataset
        dataset = pd.read_csv(dataset_path)
        if dataset.empty:
            raise ValueError("The dataset is empty. Please check the CSV file.")

        # Select only numerical columns
        numerical_columns = dataset.select_dtypes(include=[np.number]).columns.tolist()
        if not numerical_columns:
            raise ValueError("The dataset does not contain any numerical attributes for clustering.")

        # Remove columns with only one unique value (no variance)
        constant_columns = [col for col in numerical_columns if dataset[col].nunique() <= 1]
        if constant_columns:
            print(f"Removing constant columns (no variance): {constant_columns}")
            dataset.drop(columns=constant_columns, inplace=True)
        else:
            print("No constant columns detected.")

        # Update the list of numerical columns after removal
        numerical_columns = dataset.select_dtypes(include=[np.number]).columns.tolist()
        if not numerical_columns:
            raise ValueError("No numerical attributes left for clustering after removing constant columns.")

        print(f"Dataset loaded with {dataset.shape[0]} records and {dataset.shape[1]} attributes.")

        # Initialize the ClusterAgent
        agent = ClusterAgent(dataset)

        # Step 1: Calculate the correlation matrix
        agent.calculate_correlation_matrix(method='pearson')

        # Step 2: Standardize the data
        agent.standardize_data()

        # Step 3: Calculate the affinity matrix
        agent.calculate_affinity_matrix(method='cosine')

        # Step 4: Perform clustering
        cluster_method = 'spectral'       # Choose 'kmeans', 'hierarchical', or 'spectral'
        num_clusters = 5                  # Number of clusters for 'kmeans' and 'spectral'
        distance_threshold = 0.5          # Threshold for 'hierarchical' clustering

        agent.calculate_clusters(
            method=cluster_method,
            num_clusters=num_clusters,
            distance_threshold=distance_threshold,
            similarity_matrix_method='cosine'
        )

        # Step 5: Get the clustered attributes
        clustered_attributes = agent.extract_clustered_attributes()
        print("\nClustered Attributes:", clustered_attributes)

        # Step 6: Describe the clusters
        agent.describe_clusters(clustered_attributes, top_n=3)

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
