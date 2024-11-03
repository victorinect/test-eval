import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, SpectralClustering
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import cosine_similarity

class ClusterAgent:
    def __init__(self, dataset):
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

    def calculate_correlation_matrix(self, method='pearson'):
        """
        Calculate the correlation matrix for the attributes and save a heatmap.

        Parameters:
        - method: str, the type of correlation ('pearson', 'spearman').
        """
        if method not in ['pearson', 'spearman']:
            raise ValueError("Unsupported correlation method. Choose 'pearson' or 'spearman'.")

        self.correlation_matrix = self.dataset.corr(method=method)

        # Visualization: Correlation matrix heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True,
                    cbar_kws={"shrink": .5}, linewidths=.5)
        plt.title(f'{method.capitalize()} Correlation Matrix Heatmap')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        heatmap_path = os.path.join(self.visualization_dir, f'correlation_matrix_{method}.png')
        plt.savefig(heatmap_path, dpi=300)
        plt.close()
        print(f"Correlation matrix calculated and heatmap saved at {heatmap_path}.")
        return self.correlation_matrix

    def standardize_data(self):
        """
        Standardizes the data to have zero mean and unit variance.
        This is essential for clustering algorithms that are sensitive to feature scales.
        """
        scaler = StandardScaler()
        self.scaled_data = scaler.fit_transform(self.dataset)
        print("Data standardized for clustering.")

    def calculate_affinity_matrix(self, method='cosine'):
        """
        Calculate the affinity matrix (pairwise similarity matrix) from the scaled data.

        Parameters:
        - method: str, the type of similarity ('cosine', 'rbf', etc.).

        Note: Currently, only 'cosine' is implemented. Extend as needed.
        """
        if self.scaled_data is None:
            raise ValueError("Data not standardized. Run standardize_data first.")

        if method == 'cosine':
            # Compute cosine similarity between attributes (columns)
            self.affinity_matrix = cosine_similarity(self.scaled_data.T)
        else:
            raise ValueError("Unsupported affinity method. Currently only 'cosine' is implemented.")

        # Ensure the affinity matrix is non-negative
        self.affinity_matrix = np.clip(self.affinity_matrix, 0, 1)

        # Visualization: Affinity matrix heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.affinity_matrix, cmap='viridis', square=True,
                    cbar_kws={"shrink": .5}, linewidths=.5)
        plt.title(f"Affinity Matrix Heatmap ({method.capitalize()} Similarity)")
        plt.xlabel("Attributes")
        plt.ylabel("Attributes")
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        affinity_heatmap_path = os.path.join(self.visualization_dir, f'affinity_matrix_heatmap_{method}.png')
        plt.savefig(affinity_heatmap_path, dpi=300)
        plt.close()

        print(f"Affinity matrix calculated and heatmap saved at {affinity_heatmap_path}.")
        return self.affinity_matrix

    def calculate_clusters(self, method='spectral', num_clusters=3, distance_threshold=0.5, similarity_matrix_method='cosine'):
        """
        Perform clustering on attributes based on the specified method and visualize.

        Parameters:
        - method: str, the clustering method ('kmeans', 'hierarchical', 'spectral').
        - num_clusters: int, the number of clusters to form (used for 'kmeans' and 'spectral').
        - distance_threshold: float, the linkage distance threshold for 'hierarchical' clustering.
        - similarity_matrix_method: str, the similarity matrix used ('cosine', etc.).
        """
        self.cluster_method = method
        self.num_clusters = num_clusters

        if self.correlation_matrix is None:
            raise ValueError("Correlation matrix not computed. Run calculate_correlation_matrix first.")

        if method == 'kmeans':
            # For KMeans, each attribute is a data point with features as correlations with other attributes
            self.cluster_labels = KMeans(n_clusters=num_clusters, random_state=42).fit_predict(self.correlation_matrix)
        
        elif method == 'hierarchical':
            # Hierarchical clustering with distance threshold
            # Convert correlation to distance
            distance_matrix = 1 - self.correlation_matrix
            # Convert to condensed distance matrix
            condensed_distance = squareform(distance_matrix, checks=False)
            # Perform hierarchical clustering
            linkage_matrix = linkage(condensed_distance, method='ward')
            # Assign clusters based on the distance threshold
            self.cluster_labels = fcluster(linkage_matrix, t=distance_threshold, criterion='distance')

            # Visualization: Dendrogram
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
            dendrogram_path = os.path.join(self.visualization_dir, 'hierarchical_dendrogram.png')
            plt.savefig(dendrogram_path, dpi=300)
            plt.close()
            print(f"Hierarchical dendrogram saved at {dendrogram_path}.")

        elif method == 'spectral':
            if self.affinity_matrix is None:
                # If affinity matrix not computed, compute it
                self.calculate_affinity_matrix(method=similarity_matrix_method)
            spectral_clustering = SpectralClustering(n_clusters=num_clusters, affinity='precomputed', random_state=42)
            self.cluster_labels = spectral_clustering.fit_predict(self.affinity_matrix)
        
        else:
            raise ValueError("Unsupported clustering method. Choose 'kmeans', 'hierarchical', or 'spectral'.")

        # Process clusters
        self._process_clusters(method)

        # Visualization: Clustermap with clustered attributes
        self._save_clustermap()

        print(f"Clustering completed using {method} with {self.num_clusters} clusters and visualizations saved.")
        return self.cluster_labels

    def _process_clusters(self, method):
        """
        Processes the clusters to separate strongly correlated clusters from weak ones.

        Parameters:
        - method: str, the clustering method used.
        """
        clusters_dict = {}
        # Define a threshold for average intra-cluster correlation
        correlation_threshold = 0.7  # Adjust as needed

        # Iterate through each cluster
        for cluster_label in np.unique(self.cluster_labels):
            if method == 'hierarchical':
                current_label = cluster_label  # Labels start at 1
            else:
                current_label = cluster_label + 1  # Labels start at 0

            attributes = self.correlation_matrix.index[self.cluster_labels == cluster_label].tolist()
            
            if len(attributes) < 2:
                # Single attribute cannot form a cluster
                self.unclustered_attributes.extend(attributes)
                continue

            # Calculate the average correlation within the cluster
            sub_corr_matrix = self.correlation_matrix.loc[attributes, attributes]
            # Extract the upper triangle without the diagonal
            upper_tri = sub_corr_matrix.where(np.triu(np.ones(sub_corr_matrix.shape), k=1).astype(bool))
            avg_corr = upper_tri.stack().mean()

            if avg_corr >= correlation_threshold:
                clusters_dict[current_label] = attributes
            else:
                # Attributes do not meet the correlation threshold
                self.unclustered_attributes.extend(attributes)

        self.clustered_attributes = clusters_dict
        print(f"Number of strong clusters formed: {len(self.clustered_attributes)}")
        if self.unclustered_attributes:
            print(f"Number of unclustered attributes: {len(self.unclustered_attributes)}")
            print(f"Unclustered Attributes: {self.unclustered_attributes}")

    def _save_clustermap(self):
        """
        Saves a clustermap of the correlation matrix highlighting clustered attributes.
        """
        plt.figure(figsize=(12, 10))
        # Order attributes: clustered first, then unclustered
        ordered_attributes = []
        for cluster_attrs in self.clustered_attributes.values():
            ordered_attributes.extend(cluster_attrs)
        # Remove duplicates while preserving order
        ordered_attributes = list(dict.fromkeys(ordered_attributes))
        
        # If there are unclustered attributes, append them at the end
        if self.unclustered_attributes:
            ordered_attributes.extend(self.unclustered_attributes)
        
        # Ensure all attributes are included
        missing_attrs = set(self.correlation_matrix.index) - set(ordered_attributes)
        if missing_attrs:
            ordered_attributes.extend(list(missing_attrs))
        
        # Plot clustermap
        sns.clustermap(
            self.correlation_matrix.loc[ordered_attributes, ordered_attributes],
            row_cluster=True,
            col_cluster=True,
            cmap='coolwarm',
            figsize=(12, 10),
            linewidths=.5,
            annot=True,
            fmt=".2f"
        )
        plt.title("Clustermap of Strongly Correlated Attributes")
        plt.tight_layout()
        clustermap_path = os.path.join(self.visualization_dir, 'attributes_clustermap.png')
        plt.savefig(clustermap_path, dpi=300)
        plt.close()
        print(f"Clustermap of attributes saved at {clustermap_path}.")

    def extract_clustered_attributes(self):
        """
        Extracts and organizes attributes based on cluster labels.

        Returns:
        - clusters_dict: A dictionary where each key is a cluster and each value is a list of attribute names.
        """
        if not self.clustered_attributes and not self.unclustered_attributes:
            raise ValueError("Clusters not calculated. Run calculate_clusters first.")

        print("Clustered attributes extracted.")
        return self.clustered_attributes

    def describe_clusters(self, clusters_dict, top_n=3):
        """
        Describes each cluster by listing its top N attributes based on their correlation differences.

        Parameters:
        - clusters_dict: dict, dictionary mapping cluster labels to lists of attributes.
        - top_n: int, number of top attributes to list for each cluster.

        Prints the description to the console.
        """
        print("\nCluster Descriptions:")
        for cluster, attributes in clusters_dict.items():
            print(f"\nCluster {cluster}:")
            print(f"Number of Attributes: {len(attributes)}")
            print("Attributes:")
            for attr in attributes:
                print(f" - {attr}")
            # Additional insights can be added here based on domain knowledge.

        if self.unclustered_attributes:
            print("\nUnclustered Attributes:")
            for attr in self.unclustered_attributes:
                print(f" - {attr}")

def main():
    """
    Main function to execute the clustering process.
    """
    try:
        # Load the data from the CSV file
        dataset_path = "./results/underwater/style_analysis_results.csv"
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"The dataset file was not found at the path: {dataset_path}")

        dataset = pd.read_csv(dataset_path)
        if dataset.empty:
            raise ValueError("The dataset is empty. Please check the CSV file.")

        # Ensure that the dataset contains only numerical attributes for clustering
        numerical_columns = dataset.select_dtypes(include=[np.number]).columns.tolist()
        if not numerical_columns:
            raise ValueError("The dataset does not contain any numerical attributes for clustering.")

        dataset = dataset[numerical_columns]
        print(f"Dataset loaded with {dataset.shape[0]} records and {dataset.shape[1]} attributes.")

        agent = ClusterAgent(dataset)

        # Step 1: Calculate the correlation matrix (attribute-wise)
        correlation_matrix = agent.calculate_correlation_matrix(method='pearson')

        # Step 2: Standardize the data
        agent.standardize_data()

        # Step 3: Calculate affinity matrix (feature-wise similarity)
        affinity_matrix = agent.calculate_affinity_matrix(method='cosine')

        # Step 4: Calculate clusters of attributes
        # Choose 'kmeans', 'hierarchical', or 'spectral' as the method
        cluster_method = 'spectral'  # Options: 'kmeans', 'hierarchical', 'spectral'
        num_clusters = 5  # Example: 5 clusters (used for 'kmeans' and 'spectral')
        distance_threshold = 0.5  # Example threshold for 'hierarchical'

        cluster_labels = agent.calculate_clusters(
            method=cluster_method,
            num_clusters=num_clusters,
            distance_threshold=distance_threshold,
            similarity_matrix_method='cosine'
        )

        # Step 5: Extract clustered attributes
        clustered_attributes = agent.extract_clustered_attributes()
        print("\nClustered Attributes:", clustered_attributes)

        # Step 6: Describe clusters
        agent.describe_clusters(clustered_attributes, top_n=3)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
