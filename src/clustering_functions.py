from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import numpy as np


def get_cluster_kmeans(tfidf_matrix, num_clusters):
    km = KMeans(n_clusters = num_clusters)
    km.fit(tfidf_matrix)
    cluster_list = km.labels_.tolist()
    return cluster_list


def get_dbscan_cluster(tfidf_matrix, epsilon):
    db = DBSCAN(eps= epsilon, min_samples= 3).fit(tfidf_matrix)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    #n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    return labels


def multidim_scaling(similarity_matrix, n_components):
    one_min_sim = 1 - similarity_matrix
    mds = MDS(n_components=n_components, dissimilarity="precomputed", random_state=4)
    pos = mds.fit_transform(one_min_sim)  # shape (n_components, n_samples)
    x_pos, y_pos = pos[:, 0], pos[:, 1]
    return (x_pos, y_pos)


def pca_reduction(similarity_matrix, n_components):
    one_min_sim = 1 - similarity_matrix
    pca = PCA(n_components=10)
    pos = pca.fit_transform(one_min_sim)
    x_pos, y_pos = pos[:, 0], pos[:, 1]
    return (x_pos, y_pos)


def tsne_reduction(similarity_matrix):
    one_min_sim = 1 - similarity_matrix
    tsne = TSNE(learning_rate=1000).fit_transform(one_min_sim)
    x_pos, y_pos = tsne[:, 0], tsne[:, 1]
    return (x_pos, y_pos)
