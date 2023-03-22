from tslearn import clustering
from tslearn.metrics import cdist_dtw #Used to generate similarity matrix for timeseries
from tslearn.clustering import TimeSeriesKMeans
from sklearn.cluster import DBSCAN
import time


class tseries_clusterer:

    def __init__(self,data:list) -> None:
        self.data = data
        self.distance_matrix = None
    
    def kmeans(self,clusters=3,max_iter=50,random_state=0):
        labels = TimeSeriesKMeans(n_clusters=clusters,max_iter=max_iter,random_state=random_state).fit_predict(self.data)
        return labels
    
    def dbscan(self,eps=0.1,min_neighbours=3):
        start = time.time()
        if self.distance_matrix!=None:
            self.distance_matrix = cdist_dtw(self.data,self.data) #will take considerable amount of time
        end = time.time()
        print("Similarity matix precomputed, took ",(end-start),"seconds for data size",len(self.data),"\n and shape",self.data[0].shape)
        labels = DBSCAN(eps=eps,min_samples=min_neighbours,metric="precomputed").fit_predict(self.distance_matrix)
        return labels


        


