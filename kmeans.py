import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score

iris = pd.read_csv('iris.data', header=None)
X = iris.iloc[:, :-1].values

n_clusters = 3

kmeans = KMeans(n_clusters=n_clusters)

kmeans.fit(X)

labels = kmeans.labels_

db_score = davies_bouldin_score(X, labels)

print("Métrica Davies-Bouldin:", db_score)