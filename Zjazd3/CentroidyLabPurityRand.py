import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

variables = ['X', 'Y']
#labels = ['p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6']
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

#X = [[5,1], [7,1],[1,3],[7,3],[1,5],[4,5], [2,7]]
X = [[1,3], [3,3],[4,3],[5,3],[1,2],[4,2], [1,1], [2,1]]
df = pd.DataFrame(X, columns=variables, index=labels)
print(df)

km = KMeans(n_clusters=2, 
            init='random', 
            n_init=10, 
            max_iter=300,
            tol=1e-04,
            random_state=0)

y_km = km.fit_predict(X)
print(y_km)

y = [0, 1, 1, 0, 1, 1, 0, 0]
print(y)
import numpy as np
from sklearn import metrics

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    print(contingency_matrix)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 


print(purity_score(y, y_km)) # trzy klasy, w ac i dbscan sa 2 klasy


from sklearn.metrics.cluster import rand_score
print(rand_score(y, y_km))

print('Znieksztalcenie: %.2f' %km.inertia_)
print('Klastry: %s' %km.cluster_centers_)
print('Etykiety skupień: %s' %km.labels_)

#plt.scatter(km.cluster_centers_[0], km.cluster_centers_[1], edgecolors='red')  











