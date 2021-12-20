from tslearn import clustering
from tslearn.metrics import cdist_dtw #Used to generate similarity matrix for timeseries
from tslearn.clustering import TimeSeriesKMeans
from sklearn.cluster import DBSCAN


class tseries_clusterer:

    def __init__(self,data:list) -> None:
        self.data = data
        self.distance_matrix = None
    


    def kmeans(self,clusters=3,max_iter=50,random_state=0):
        km = TimeSeriesKMeans(n_clusters=clusters,max_iter=max_iter,random_state=random_state).fit(self.data)
        return km._labels
    
    def dbscan(self,eps=0.1,min_neighbours=3):
        self.distance_matrix = cdist_dtw(self.data,self.data) #will take considerable amount of time
        dbs = DBSCAN(eps=eps,min_samples=min_neighbours,metric="precomputed").fit(self.distance_matrix)
        return dbs._labels




