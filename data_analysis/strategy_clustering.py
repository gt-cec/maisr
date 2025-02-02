import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from pathlib import Path


def load_and_preprocess_data(file_path):
    """
    Load data from Excel file and preprocess it for clustering.

    Args:
        file_path (str): Path to the Excel file

    Returns:
        tuple: (preprocessed_data, original_data, feature_names)
    """
    # Read Excel file
    data = pd.read_excel(file_path)

    # Store feature names
    feature_names = data.columns.tolist()

    # Scale the features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    return scaled_data, data, feature_names


def find_optimal_clusters(data, max_clusters=10):
    """
    Find optimal number of clusters using silhouette score.

    Args:
        data (numpy.ndarray): Preprocessed data
        max_clusters (int): Maximum number of clusters to try

    Returns:
        int: Optimal number of clusters
    """
    silhouette_scores = []

    # Try different numbers of clusters
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_scores.append(silhouette_avg)

    # Find optimal number of clusters
    optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2

    # Plot silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs Number of Clusters')
    plt.savefig('silhouette_scores.png')
    plt.close()

    return optimal_clusters


def perform_kmeans_clustering(data, n_clusters):
    """
    Perform K-means clustering.

    Args:
        data (numpy.ndarray): Preprocessed data
        n_clusters (int): Number of clusters

    Returns:
        numpy.ndarray: Cluster labels
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit_predict(data)


def perform_hierarchical_clustering(data, feature_names):
    """
    Perform hierarchical clustering and create dendrogram.

    Args:
        data (numpy.ndarray): Preprocessed data
        feature_names (list): List of feature names
    """
    # Create linkage matrix
    linkage_matrix = linkage(data, method='ward')

    # Plot dendrogram
    plt.figure(figsize=(12, 8))
    dendrogram(linkage_matrix)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.savefig('dendrogram.png')
    plt.close()


def analyze_clusters(data_df, cluster_labels, feature_names):
    """
    Analyze characteristics of each cluster.

    Args:
        data_df (pandas.DataFrame): Original data
        cluster_labels (numpy.ndarray): Cluster assignments
        feature_names (list): List of feature names

    Returns:
        pandas.DataFrame: Cluster analysis results
    """
    data_df['Cluster'] = cluster_labels
    cluster_analysis = data_df.groupby('Cluster').agg(['mean', 'std'])

    # Format the analysis for better readability
    analysis_results = []
    for cluster in range(len(set(cluster_labels))):
        cluster_data = {
            'Cluster': cluster,
            'Size': sum(cluster_labels == cluster),
            'Distinctive Features': []
        }

        # Find distinctive features (those where the cluster mean differs significantly from overall mean)
        for feature in feature_names:
            cluster_mean = data_df[data_df['Cluster'] == cluster][feature].mean()
            overall_mean = data_df[feature].mean()
            overall_std = data_df[feature].std()

            if abs(cluster_mean - overall_mean) > overall_std:
                difference = cluster_mean - overall_mean
                cluster_data['Distinctive Features'].append(
                    f"{feature}: {difference:+.2f} from mean"
                )

        analysis_results.append(cluster_data)

    return pd.DataFrame(analysis_results)


def main(file_path):
    """
    Main function to run the clustering analysis.

    Args:
        file_path (str): Path to the Excel file
    """
    # Load and preprocess data
    scaled_data, original_data, feature_names = load_and_preprocess_data(file_path)

    # Find optimal number of clusters
    optimal_clusters = find_optimal_clusters(scaled_data)
    print(f"Optimal number of clusters: {optimal_clusters}")

    # Perform K-means clustering
    cluster_labels = perform_kmeans_clustering(scaled_data, optimal_clusters)

    # Perform hierarchical clustering
    perform_hierarchical_clustering(scaled_data, feature_names)

    # Analyze clusters
    cluster_analysis = analyze_clusters(original_data, cluster_labels, feature_names)

    # Save results
    print("\nCluster Analysis:")
    print(cluster_analysis.to_string())

    # Save cluster assignments to Excel
    original_data['Cluster'] = cluster_labels
    original_data.to_excel('clustered_results.xlsx', index=False)

    print("\nResults have been saved:")
    print("1. clustered_results.xlsx - Original data with cluster assignments")
    print("2. silhouette_scores.png - Plot of silhouette scores")
    print("3. dendrogram.png - Hierarchical clustering dendrogram")


if __name__ == "__main__":
    # Replace with your Excel file path
    file_path = "strategy_data.xlsx"
    main(file_path)