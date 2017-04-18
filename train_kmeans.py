from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import sklearn.metrics
import matplotlib
matplotlib.use("agg")
from preprocess import ClimateData
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main():
    climatology = ClimateData('/gss_gpfs_scratch/vandal.t/data-mining-project')
    d = 'historical'
    data = getattr(climatology,d)
    k = 9
    best_dist, best_model = -1, None
    for k in range(k,k+1):
        model = Pipeline([('normalize', StandardScaler()), 
                          ('kmeans', KMeans(n_clusters=k, n_init=20, n_jobs=4))])
        idxs = np.random.choice(range(data.x.shape[0]), 50000)
        model.fit(data.x[idxs])
        labels_ = model.predict(data.x)
        dist = sklearn.metrics.silhouette_score(data.x[idxs], labels_[idxs])
        print "K=%i, Dist=%f" % (k, dist)
        if dist > best_dist:
            best_dist = dist
            best_model = model
    for d in ['historical', 'rcp45', 'rcp85']:
        data = getattr(climatology, d)
        labels_ = best_model.predict(data.x)
        dist = sklearn.metrics.silhouette_score(data.x[idxs], labels_[idxs])
        print d, "Dist=", dist
        res = np.concatenate([data.latlon, labels_[:,np.newaxis]], axis=1)
        df = pd.DataFrame(res, columns=['lat', 'lon', 'membership'])
        df = pd.pivot_table(df, values='membership', columns='lon', index='lat')
        df.to_csv("output/kmeans_%s.csv" % d)
        plt.imshow(df.values[::-1], cmap=plt.cm.Accent)
        plt.title("%s --- Silhouette Score: %0.2f" % (d, dist))
        plt.savefig("figures/kmeans_%s.pdf" % d)
        #plt.show()

if __name__ == "__main__":
    main()
