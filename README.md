# timeseries_clustering
Time series clustering using different algos. Its a wrapper 

Usage 
```python
import numpy as np
from tseries_clustering import tseries_clusterer

data = [ ndarray1, ndarray2,.....] -> list #ndarray is a time_series of shape time_steps x features

clusterer = tseries_clusterer(data)
#labels = clusterer.kmeans()
labels = clusterer.dbscan()
```
