import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np

X = np.array(([[1,2],
               [1.5,1.8],
               [5,8],
               [8,8],
               [1,0.6],
               [9,11],
               [1,3],
                     [8,9],
                     [0,3],
                     [5,4],
                     [7,8],
                     [6,4],]))

plt.scatter(X[:,0], X[:,1], s=150, linewidths=5)
plt.show()

colors = 10 * ['g', 'r', 'c', 'b', 'k']


class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter= 300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):
        
        self.centriods = {}

        for i in range(self.k):
            # initialize the centriods postion as first data points
            # pick starting centers
            self.centriods[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {}
            # classifi[cluster] = [points]
            for i in range(self.k):
                self.classifications[i] = []
            # cycle through know data sets , and assign class it is closest to
            for featureset in data :
                distances = [np.linalg.norm(featureset - self.centriods[centriod]) for centriod in self.centriods]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)
            
            prev_centriods = dict(self.centriods)

            for classification in self.classifications:
                self.centriods[classification] = np.average(self.classifications[classification], axis=0)

            optimized = True

            for c in self.centriods:
                original_centriod = prev_centriods[c]
                current_centriod = self.centriods[c]
                if np.sum((current_centriod - original_centriod)/ original_centriod * 100.0) > self.tol:
                    optimized = False
                
            if optimized :
                break



    def predict(self, data):
        distances = [np.linalg.norm(data - self.centriods[centriod]) for centriod in self.centriods]
        classification = distances.index(min(distances))
        return classification



clf = K_Means()
clf.fit(X)

for centriod in clf.centriods:
    plt.scatter(clf.centriods[centriod][0], clf.centriods[centriod][1],
                marker='o', color='k', s=150, linewidths=5)

for classification in  clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker='x', c=color, s=150, linewidths=5)

unknowns = np.array([[1,3],
                     [8,9],
                     [0,3],
                     [5,4],
                     [7,8],
                     [6,4], ])

for unknown in unknowns:
    classification = clf.predict(unknown)
    plt.scatter(unknown[0], unknown[1], marker='*', c=colors[classification], s=150, linewidths=5)

plt.show()