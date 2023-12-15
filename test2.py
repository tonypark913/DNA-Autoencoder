import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

def calculate_keyword_proportions(data, clusters, keywords):
    cluster_keyword_proportions = {}
    for cluster in np.unique(clusters):
        cluster_data = data[clusters == cluster]
        keyword_counts = {keyword: 0 for keyword in keywords}
        for index, row in cluster_data.iterrows():
            for column in data.columns[2:20]:  # Classification 1~18까지 순회
                for keyword in keywords:
                    if isinstance(row[column], str) and keyword in row[column]:
                        keyword_counts[keyword] += 1
        keyword_proportions = {keyword: count / len(cluster_data) for keyword, count in keyword_counts.items()}
        cluster_keyword_proportions[cluster] = keyword_proportions
    return cluster_keyword_proportions

def plot_keywords(ax, data, pca_data, keywords, colors):
    color_dict = dict(zip(keywords, colors))
    keyword_added = {keyword: False for keyword in keywords}
    for index, row in data.iterrows():
        color = 'black'
        label = None
        for column in data.columns[2:20]:  # Classification 1~18까지 순회
            for keyword in keywords:
                if isinstance(row[column], str) and keyword in row[column]:
                    color = color_dict[keyword]
                    if not keyword_added[keyword]:
                        label = keyword
                        keyword_added[keyword] = True
                    break  # 키워드를 찾았으므로 다음 행으로 이동
            if color != 'black':
                break
        data_point = pca_data[index]
        ax.scatter(data_point[0], data_point[1], s=15, color=color, label=label, picker=True)
    ax.legend()

def plot_pca_scatter(data_file, eps, min_samples, eps1, min_samples1, keyword):
    """
    Perform PCA for dimensionality reduction on the given data file and create a scatter plot of the reduced data.
    If the keyword is found in any of the Classification columns, the data point will be plotted with a different color.

    Args:
        data_file (str): The path to the CSV file containing the data.
        eps (float): The maximum distance between two samples for them to be considered as in the same neighborhood.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
        eps1 (float): The maximum distance between two samples for them to be considered as in the same neighborhood for noise data.
        min_samples1 (int): The number of samples in a neighborhood for a point to be considered as a core point for noise data.
        keyword (str): The keyword to search for in the Classification columns.

    Returns:
        None
    """

    # Create a new figure with two subplots
    fig, axs = plt.subplots(1, 3, figsize=(15,15))
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        #ax.set_facecolor('#eeeeee')
    data = pd.read_csv(data_file)
    # gs = gridspec.GridSpec(3, 2, width_ratios=[3, 1], height_ratios=[2, 1]) 
    ax0 = fig.add_subplot(131, projection='3d')
    ax0.set_title('Raw Data')
    ax0.scatter(data['Encoded_x'], data['Encoded_y'], data['Encoded_z'], s=15, picker=True)
    ax0.patch.set_linewidth(2)  # Set border thickness
    ax0.patch.set_edgecolor('black')  # Set border
    # Call the function to plot the scatter plot on the left subplot
    ax1 = fig.add_subplot(322)
    ax1.set_title('DBSCAN Clustering')
    ax1.text(0.05, 0.95, f'eps: {eps}, min_samples: {min_samples}', ha='left', va='top', transform=ax1.transAxes, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
    # Call the function to plot the scatter plot on the right subplot
    ax2 = fig.add_subplot(324)
    ax2.set_title('Keyword Search')

    ax3 = fig.add_subplot(313)
    ax3.set_title('Keyword Proportions')
    fig.suptitle('tRNA-ALA Encoded Data', fontsize=16, y=0.95, weight='bold')
        # Load the data from the CSV file
    data1 = pd.read_csv("tRNA-ALA.csv")
    # fig.set_facecolor('#eeeeee')

    # Perform PCA for dimensionality reduction
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data[['Encoded_x', 'Encoded_y', 'Encoded_z']])

    # Perform clustering on the reduced data
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(data_pca)

    noise_data = data_pca[clusters == -1]

    # Perform DBSCAN on the noise data
    dbscan_noise = DBSCAN(eps=eps1, min_samples=min_samples1)
    clusters_noise = dbscan_noise.fit_predict(noise_data)

    # Add the new clusters to the original clusters
    #max_cluster = max(clusters)
    #clusters_noise[clusters_noise != -1] += max_cluster + 1
    
    #clusters[clusters == -1] = clusters_noise

    #noise_data = data_pca[clusters == -1]

    # Perform DBSCAN on the noise data
    #dbscan_noise = DBSCAN(eps=0.2, min_samples=6)
    #clusters_noise = dbscan_noise.fit_predict(noise_data)

    #max_cluster = max(clusters)
    #clusters_noise[clusters_noise != -1] += max_cluster + 1

    # Add the new clusters to the original clusters
    #clusters[clusters == -1] = clusters_noise

    unique_clusters = np.unique(clusters)
    colors = plt.cm.hsv(np.linspace(0, 1, len(unique_clusters)))

    # Create a scatter plot of the reduced data with different colors for each cluster
    unique_clusters = set(clusters)
    for cluster, colors in zip(unique_clusters, colors):
        cluster_data = data_pca[clusters == cluster]
        if(cluster==-1):
            ax1.scatter(cluster_data[:, 0], cluster_data[:, 1], s=15, color='black', label=f'Cluster {cluster}', picker=True)
        else: 
            ax1.scatter(cluster_data[:, 0], cluster_data[:, 1], s=15, color=colors, label=f'Cluster {cluster}', picker=True)
    ax1.legend()  # 범례 출력
    # Set the labels for the plot
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    keyword_added = False

    plot_keywords(ax2, data1, data_pca, ['Mammalia', 'Rhodophyta', 'Actinopterygii'], ['blue', 'red', 'orange'])
    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys(), loc='upper right') # 범례 출력
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    keyword_proportions = calculate_keyword_proportions(data1, clusters, ['Rhodophyta', 'Mammalia', 'Actinopterygii'])
    for keyword in keyword_proportions[next(iter(keyword_proportions))].keys():
        proportions = [proportions[keyword] for proportions in keyword_proportions.values()]
        clusters = list(keyword_proportions.keys())
        ax3.bar(clusters, proportions, label=f'Keyword {keyword}')

    ax3.legend()


    # Add the 'Scientific name' as labels for each data point
    # for i, name in enumerate(data['Scientific name']):
    #     plt.annotate(name, (data_pca[i, 0], data_pca[i, 1]), fontsize=8, xytext=(2, 1), textcoords='offset points')

    # Search for the keyword in the Classification columns
    # Save the plot as a CSV file

    # Show the legend
    plt.legend()

    # plt.savefig('output.png', dpi=300)

    # Show the plot
    plt.show()

plot_pca_scatter('tRNA-ALA-encoded.csv', eps=0.015, min_samples=3, eps1=0.07, min_samples1=10, keyword='Homo')
