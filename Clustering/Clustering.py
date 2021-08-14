import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation, SpectralClustering
import numpy as np
import pickle
import plotly as py
import plotly.tools as tls
import cufflinks as cf
import plotly.graph_objects as go
py.offline.init_notebook_mode(connected = True)
cf.go_offline()
import plotly.express as px
from sklearn import metrics
import pandas as pd

from Logging.Logger import Logger


class Clustering:
    def __init__(self):
        pass

    def fit(self, x_train, x_test):
       try:
           L = ["K-Means",  'AffinityPropagation', 'SpectralClustering' , 'AgglomerativeClustering']
           Score = []
           Model = []
           data = []

           ## K-Means
           wcss = []
           for i in range(1, 11):
               kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)  # initializing the KMeans object
               kmeans.fit(x_train)  # fitting the data to the KMeans Algorithm
               wcss.append(kmeans.inertia_)

           fig = px.line(x=range(1, 11), y=wcss,
                         labels={'x': "Number of clusters", 'y': "WCSS"},
                         title="K-Means : The Elbow Method")

           self.kn = KneeLocator(range(1, 11), wcss, curve='convex', direction='decreasing')
           # print(self.kn.knee)

           fig.add_vrect(x0=self.kn.knee, x1=self.kn.knee)
           fig.write_image("result/K-Means_Elbow_Plotly.png")
           fig.show()

           kmeans = KMeans(n_clusters=self.kn.knee, init='k-means++', random_state=42)
           kmeans.fit_transform(x_train)
           labels = kmeans.fit_predict(x_test)

           Score.append(metrics.silhouette_score(x_test, labels))

           with open('result/kmeans.pickle', 'wb') as f:
               pickle.dump(kmeans, f)
               Logger().log("Clustering.py", "INFO",
                            "Saved kmeans_Model : as kmeans.pickle")


           data.append(({
               "Silhouette_Score": metrics.silhouette_score(x_test, labels) ,
               'calinski_harabasz_score': metrics.calinski_harabasz_score(x_test, labels),
               'davies_bouldin_score': metrics.davies_bouldin_score(x_test, labels),
               "Model_Download": '<a href="//result/kmeans.pickle" download="result/kmeans.pickle">kmeans  </a>'

           }))
           Logger().log("Clustering.py", "INFO", "K-Means is performed on the Dataset ")
           Logger().log("Clustering.py", "INFO", "Saved K-Means Model as K-Means.pickle")

           algorithms = []
           algorithms.append(AffinityPropagation())
           algorithms.append(SpectralClustering(n_clusters=10, random_state=1,
                                                affinity='nearest_neighbors'))
           algorithms.append(AgglomerativeClustering(n_clusters=10))

           for i in range(len(algorithms)):
               model = algorithms[i].fit(x_train)
               labels = model.fit_predict(x_test)
               Score.append(metrics.silhouette_score(x_test, labels))
               with open('result/'+ L[i+1] +'.pickle', 'wb') as f:
                   pickle.dump(model, f)
               data.append(({
                   "Silhouette_Score": metrics.silhouette_score(x_test, labels),
                   'calinski_harabasz_score': metrics.calinski_harabasz_score(x_test, labels),
                   'davies_bouldin_score': metrics.davies_bouldin_score(x_test, labels),
                   "Model_Download": '<a href="//result/' + L[i+1] + '.pickle" download="result/' + L[i+1] + '.pickle">'+ L[i+1] +'</a>'

               }))




           Logger().log("Clustering.py", "INFO", "AffinityPropagation is performed on the Dataset ")
           Logger().log("Clustering.py", "INFO", "Saved AffinityPropagation Model as AffinityPropagation.pickle")
           Logger().log("Clustering.py", "INFO", "SpectralClustering is performed on the Dataset ")
           Logger().log("Clustering.py", "INFO", "Saved SpectralClustering Model as SpectralClustering.pickle")
           Logger().log("Clustering.py", "INFO", "AgglomerativeClustering is performed on the Dataset ")
           Logger().log("Clustering.py", "INFO", "Saved AgglomerativeClustering Model as AgglomerativeClustering.pickle")

           m = max(Score)
           max_index = Score.index(m)
           best_algo = L[max_index]

           recommended = []
           for i in L:
               if i == best_algo:
                   recommended.append("Highly Recommended")
               else:
                   recommended.append(" ")

           result = pd.DataFrame(data=data, columns=['Silhouette_Score', 'calinski_harabasz_score', 'davies_bouldin_score','Model_Download'],
                                  index=['K-means', 'Affinity',
                                         'Spectral', 'Agglomerative'])

           results = pd.concat([result, pd.DataFrame(recommended, columns=["Recommended"],
                                                     index=['K-means', 'Affinity',
                                         'Spectral', 'Agglomerative'])], axis=1)
           return results
       except Exception as e:
           Logger().log("Clustering.py", "ERROR","Error while performing Clustering using K-means,Affinity,Spectral,Agglomerative"+ str(e))
           raise Exception("Error while performing Clustering using K-means,Affinity,Spectral,Agglomerative" + str(e))




