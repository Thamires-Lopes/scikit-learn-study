import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score

df = pd.read_excel('Dry_Bean_Dataset.xlsx')
X = df.iloc[:, :-1].values

# Utilizando igual a foi feito no ML.NET
n_clusters = 5

kmeans = KMeans(n_clusters=n_clusters)

kmeans.fit(X)

labels = kmeans.labels_

db_score = davies_bouldin_score(X, labels)

print("Métrica Davies-Bouldin:", db_score)