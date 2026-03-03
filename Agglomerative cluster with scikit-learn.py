import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_iris
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import adjusted_rand_score

iris = load_iris()
X = iris.data  
y = iris.target 

feature_names = iris.feature_names
linked = linkage(X, 'ward')
plt.figure(figsize=(10, 7))
dendrogram(linked,orientation='top',labels=None,  distance_sort='descending',show_leaf_counts=True) 
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Data Points')
plt.ylabel('Distance')
plt.show()

n_clusters = 3 
agg_clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
cluster_labels = agg_clustering.fit_predict(X)
iris_df = pd.DataFrame(data=X, columns=feature_names)
iris_df['cluster'] = cluster_labels
iris_df['target'] = y 
plt.figure(figsize=(8, 6))

for i in range(n_clusters):
    cluster_data = iris_df[iris_df['cluster'] == i]
    plt.scatter(cluster_data[feature_names[0]], cluster_data[feature_names[1]], label=f'Cluster {i}')
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.title('Agglomerative Clustering of Iris Dataset')
plt.legend()
plt.show()
ari = adjusted_rand_score(y, cluster_labels) 
print(f"Adjusted Rand Index: {ari}")