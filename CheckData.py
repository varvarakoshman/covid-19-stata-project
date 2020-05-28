import functools
import operator
import re
from nltk.cluster import KMeansClusterer
from sklearn import cluster
from sklearn import metrics
import nltk
from sklearn.cluster import DBSCAN
import seaborn as sns

import pandas as pd
import math
from gensim.models import Word2Vec
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn import decomposition
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.manifold import TSNE


date_cols = ['date']
# global_data_movements = pd.read_csv("data/Global_Mobility_Report.csv", dtype={"country_region_code": object, "country_region": object, "sub_region_1": object, "sub_region_2": object, "grocery_and_pharmacy_percent_change_from_baseline": float, "parks_percent_change_from_baseline": float, "transit_stations_percent_change_from_baseline": float, "workplaces_percent_change_from_baseline": float, "residential_percent_change_from_baseline": float}, parse_dates=date_cols)
# print(global_data.head())
# usa = global_data_movements[global_data_movements.country_region_code == 'US']
# print("movements details in USA: ", usa.shape[0])

# print(usa['sub_region_1'].isnull().sum())
# print(usa['sub_region_2'].isnull().sum())
# print(usa.head())

# TODO: LOAD DATA ABOUT ILLNESS DAILY STATISTICS
df = pd.read_csv("data/2020-03-12_2020-04-15_us_tweets_with_data.csv")  # , dtype={"status_id"})
# print(tweets_hashtags.head())
users = df['user_id'].tolist()
user_freq = defaultdict(int)
for user in users:
    user_freq[user] += 1
num = set(users)

sorted_freq_users = sorted(user_freq, key=user_freq.get, reverse=True)
with open('data/frequenciesusers.txt', 'w') as filehandle:
    for key in sorted_freq_users:
        filehandle.write("%s: %s\n" % (key, user_freq[key]))
hashtags = df['hashtags'].tolist()
print("# all hashtags (inlined)=", len(hashtags))

naive_unique = set(hashtags)
strings = [tag for tag in naive_unique if type(tag) != float] # nan removed
splitted = list(map(lambda hashtag: hashtag.split(), strings))
flat_mapped = functools.reduce(operator.iconcat, splitted, [])

print("# all hashtags (separated)=", len(flat_mapped))
unique = set(flat_mapped)

covid = [i for i in unique if i.find('covid') != -1 or i.find('Covid') != -1]


# with open('data/covid.txt', 'w') as filehandle:
#     for listitem in covid:
#         filehandle.write('%s\n' % listitem)
#
# with open('data/listfile.txt', 'w') as filehandle:
#     for listitem in unique:
#         filehandle.write('%s\n' % listitem)
print("# unique hashtags=", len(unique))

# tweet_text_edited = tweets_hashtags['text_edited'].tolist()
# tweet_text_edited_split = [re.split(' |_', i) for i in tweet_text_edited]

# =================
# frequencies
word_freq = defaultdict(int)
strings_2 = [tag for tag in hashtags if type(tag) != float]  # nan removed
splitted_2 = list(map(lambda hashtag: hashtag.split(), strings_2))
flat_mapped_2 = functools.reduce(operator.iconcat, splitted_2, [])
for tag in flat_mapped_2:
    word_freq[tag] += 1
len(word_freq)
sorted_freq_hashtags = sorted(word_freq, key=word_freq.get, reverse=True)
with open('data/frequencies.txt', 'w') as filehandle:
    for key in sorted_freq_hashtags:
        filehandle.write("%s: %s\n" % (key, word_freq[key]))
hashtags_not_rare = [i for i in sorted_freq_hashtags if 10 < word_freq[i] < 10000]

# my creepy plot of frequencies
# xs = [i for i in range(len(sorted_freq_hashtags))]
# ys = [i for i in word_freq.values()]
# plt.plot(xs, ys, 'bo')
# plt.show()
# =================
# word2vec
long_tags_seq = [i for i in strings if len(i.split()) > 1]
with open('data/longTagSeqs.txt', 'w') as filehandle:
    for key in long_tags_seq:
        filehandle.write("%s\n" % key)
long_tags_seq_split = [i.split() for i in long_tags_seq]
model = Word2Vec(long_tags_seq_split, min_count=1)
# vocabulary = model.wv.vocab
# vocab_embed = {}
# my_dict2 = dict((y,x) for x,y in vocabulary.iteritems())
# for i in long_tags_seq_split:
#     vocab_embed[model[i]] = i

# =================
# clustering
embeddings = [model[i] for i in hashtags_not_rare]
# long_tags_seq_split_flatten=functools.reduce(operator.iconcat, long_tags_seq_split, [])

# check optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = cluster.KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(embeddings)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title('Elbrow method')
plt.xlabel('# clusters')
plt.ylabel('wcss')
plt.show()

NUM_CLUSTERS = 5
kmeans_model = cluster.KMeans(n_clusters=NUM_CLUSTERS, init='k-means++')
labels = kmeans_model.fit_predict(embeddings)

embeddings = np.array(embeddings)
pca = decomposition.PCA(n_components=2).fit(embeddings)
coordinates = pca.transform(embeddings)
label_colors = ['#7FFF00', '#B22222', '#FFD700', '#6A5ACD', '#008080']
colors = [label_colors[i] for i in labels]
plt.scatter(coordinates[:, 0], coordinates[:, 1], c=colors)
centroids = kmeans_model.cluster_centers_
centroid_coords = pca.transform(centroids)
# plt.scatter(centroid_coords[:, 0], centroid_coords[:, 1], marker='X', s=100, linewidths=1, c='#696969')
# plt.show()
# closest, _ = pairwise_distances_argmin_min(centroids, embeddings)

labels2 = kmeans_model.labels_
silhouette_score = metrics.silhouette_score(embeddings, labels2, metric='euclidean')
print("Silhouette_score: ", silhouette_score)

def pca_scatter(coordinates, classifs):
    bar = pd.DataFrame(list(zip(coordinates[:, 0], coordinates[:, 1], classifs)), columns=["PC1", "PC2", "Class"])
    sns.lmplot("PC1", "PC2", bar, hue="Class", fit_reg=False)
    plt.show()

pca_scatter(coordinates, labels)

with open('data/group1.txt', 'w') as f1, open('data/group2.txt', 'w') as f2, open('data/group3.txt', 'w') as f3, open('data/group4.txt', 'w') as f4, open('data/group5.txt', 'w') as f5:
    index = 0
    for word in hashtags_not_rare:
        if labels[index] == 0:
            f1.write("%s\n" % word)
        elif labels[index] == 1:
            f2.write("%s\n" % word)
        elif labels[index] == 2:
            f3.write("%s\n" % word)
        elif labels[index] == 3:
            f4.write("%s\n" % word)
        elif labels[index] == 4:
            f5.write("%s\n" % word)
        index = index + 1

# model.build_vocab(tweet_text_edited)
# print (list(model.corpus_total_words))
print("hooray")