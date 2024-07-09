from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def perform_kmeans(df):
    ks, inertias = [], []

    for k in range(1, 11):
        # instantiate
        model = KMeans(n_clusters=k, n_init='auto')
    # fit
        model.fit(df)
    # score
        inertias.append(model.inertia_)
        ks.append(k)
    return ks, inertias


def fit_agglomerative_clustering(df):
    ks, sil_scores = [], []
    for k in range(2, 11):
        model = AgglomerativeClustering(n_clusters=k)
        model.fit(df)
        sil_score = silhouette_score(df, model.labels_)
        sil_scores.append(sil_score)
        ks.append(k)

    return ks, sil_scores


def plot_scores(k_range, scores, score_type):
    plt.figure()
    plt.plot(k_range, scores, marker="X")
    plt.title(f"{score_type} as a function of K (number of clusters)")
    plt.ylabel(score_type)
    plt.xlabel('k')
    plt.show()


def fit_kmeans(df, num_clust):
    final_kmeans = KMeans(n_clusters=num_clust, n_init='auto')
    final_kmeans.fit(df)
    return final_kmeans


def perform_tsne(df):
    # Instantiate TSNE
    tsne = TSNE()

    # Fit TSNE
    tsne_data = tsne.fit_transform(df)

    return tsne_data


def visualize_clusters(tsne_data, final_kmeans, final_agglom):
    # Visualize clusters using TSNE
    plt.subplots(1, 2, figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=(
        ~final_kmeans.labels_.astype(bool)).astype(int))
    plt.title("kmeans")

    plt.subplot(1, 2, 2)
    plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=final_agglom.labels_)
    plt.title("agglom")

    plt.show()
