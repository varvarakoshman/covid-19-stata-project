import functools
import operator
from sklearn import cluster
from sklearn import metrics
from sklearn.cluster import DBSCAN
import numpy as np

import pandas as pd
from gensim.models import Word2Vec
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn import decomposition
import seaborn as sns


def find_in_cluster(cluster, target):
    values = cluster.values
    for i in range(cluster.shape[0]):
        if target == values[i][0].tolist():
            return True
    return False


def compute_distance(centroid, target):
    return np.linalg.norm(np.array(target) - np.array(centroid))


def pca_scatter(coordinates, classifs):
    bar = pd.DataFrame(list(zip(coordinates[:, 0], coordinates[:, 1], classifs)), columns=["PC1", "PC2", "Class"])
    sns.lmplot("PC1", "PC2", bar, hue="Class", fit_reg=False)
    plt.show()


def main():
    df = pd.read_csv("data/2020-03-12_2020-04-15_us_tweets_with_data.csv")
    hashtags = df['hashtags'].tolist()
    naive_unique = set(hashtags)
    strings = [tag for tag in naive_unique if type(tag) != float] # nan removed

    # word2vec
    long_tags_seq = [i for i in strings if len(i.split()) > 1]
    long_tags_seq_split = [i.split() for i in long_tags_seq]
    model = Word2Vec(long_tags_seq_split, min_count=1)

    # clustering

    # 1) clustering not rare hashtags (3 < occurencies < 10k)

    # word_freq = defaultdict(int)
    # strings_2 = [tag for tag in hashtags if type(tag) != float]  # nan removed
    # splitted_2 = list(map(lambda hashtag: hashtag.split(), strings_2))
    # flat_mapped_2 = functools.reduce(operator.iconcat, splitted_2, [])
    # for tag in flat_mapped_2:
    #     word_freq[tag] += 1
    # sorted_freq_hashtags = sorted(word_freq, key=word_freq.get, reverse=True)
    # hashtags_not_rare = [i for i in sorted_freq_hashtags if 3 < word_freq[i] < 10000]

    # 2) clustering hashtags that appear everyday
    df2 = pd.read_csv("data/us_dates_to_everyday_hashtags.csv")
    hashtags_not_rare = df2['hashtags'].tolist()
    embeddings = [model[i] for i in hashtags_not_rare]

    # 3) check optimal number of clusters
    wcss = []
    for i in range(1, 11):
        kmeans = cluster.KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(embeddings)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 11), wcss)
    plt.title('Elbrow method')
    plt.xlabel('# clusters')
    plt.ylabel('wcss')
    plt.show()

    NUM_CLUSTERS = 7
    kmeans_model = cluster.KMeans(n_clusters=NUM_CLUSTERS, init='k-means++')
    labels = kmeans_model.fit_predict(embeddings)
    cluster_map = pd.DataFrame()
    cluster_map['data_index'] = embeddings
    cluster_map['cluster'] = kmeans_model.labels_
    # cluster1 = cluster_map[cluster_map.cluster == 0]
    # cluster2 = cluster_map[cluster_map.cluster == 1]
    # cluster3 = cluster_map[cluster_map.cluster == 2]
    # cluster4 = cluster_map[cluster_map.cluster == 3]
    # cluster5 = cluster_map[cluster_map.cluster == 4]
    centroids = kmeans_model.cluster_centers_
    centroid1 = np.array(centroids[0])
    centroid2 = np.array(centroids[1])
    centroid3 = np.array(centroids[2])
    centroid4 = np.array(centroids[3])
    centroid5 = np.array(centroids[4])
    centroid6 = np.array(centroids[5])
    centroid7 = np.array(centroids[6])
    all_centroids = [centroid1, centroid2, centroid3, centroid4, centroid5, centroid6, centroid7]

    tweets_dict = {} # dict: {hashtag: (cluster, dist(hashtag, centroid))}
    for hashtag in hashtags_not_rare:
        tempd = {}
        for ind, centroid in enumerate(all_centroids):
            tempd[compute_distance(centroid, np.array(model[hashtag]))] = ind
        min_dist = np.amin(list(tempd.keys()))
        cluster_n = tempd[min_dist]
        tweets_dict[hashtag] = tuple([cluster_n, min_dist])

    # clustering
    #
    # df1 = df.iloc[:, 23]
    # tweets_dict = {}
    # predicate = lambda tag_seq: type(tag_seq) != float and len(tag_seq.split()) > 1
    # for index in range(df1.shape[0]):
    #     if predicate(df1.get(index)):
    #         curr_list = df1.get(index).split()
    #         matrix = np.zeros((30, 5))
    #         for word_ind, word in enumerate(curr_list):
    #             tempd = {}
    #             for ind, centroid in enumerate(all_centroids):
    #                 tempd[compute_distance(centroid, np.array(model[word]))] = ind
    #             min_dist = np.amin(list(tempd.keys()))
    #             cluster_n = tempd[min_dist]
    #             matrix[word_ind][cluster_n] = min_dist
    #         temp_dict = {}
    #         for i in range(matrix.shape[1]):
    #             avg = np.mean(list(filter(lambda n: n > 0, matrix[:, i])))
    #             temp_dict[avg] = i
    #         nonnan = [j for j in list(temp_dict.keys()) if not np.isnan(j)]
    #         if (np.sum(nonnan) > 0):
    #             minimum = np.amin(nonnan)
    #             cluster_label_for_tweet = temp_dict[minimum]
    #             tweets_dict[index] = cluster_label_for_tweet

    with open('data/cluster_labels_for_tweets.txt', 'w') as filehandle:
        for key, value in tweets_dict.items():
            filehandle.write("%s: %s\n" % (key, value))

    pca = decomposition.PCA(n_components=2).fit(embeddings)
    coordinates = pca.transform(embeddings)
    # label_colors = ['#7FFF00', '#B22222', '#FFD700', '#6A5ACD', '#008080', '#00ffc8']
    # colors = [label_colors[i] for i in labels]
    # plt.scatter(coordinates[:, 0], coordinates[:, 1], c=colors)
    # centroids = kmeans_model.cluster_centers_
    # centroid_coords = pca.transform(centroids)
    # plt.scatter(centroid_coords[:, 0], centroid_coords[:, 1], marker='X', s=100, linewidths=1, c='#696969')
    # plt.show()
    pca_scatter(coordinates, labels)

    labels2 = kmeans_model.labels_
    silhouette_score = metrics.silhouette_score(embeddings, labels2, metric='euclidean')
    print("Silhouette_score: ", silhouette_score)


if __name__ == '__main__':
    main()