import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np

from sklearn.datasets.samples_generator import make_blobs

X, y = make_blobs(n_samples=20, centers=3, n_features=2)

# X = np.array(([[1,2],
#                [1.5,1.8],
#                [5,8],
#                [8,8],
#                [1,0.6],
#                [9,11],
#                [1,3],
#                [8,2],
#                [10,2],
#                [9,3],
#                      ]))

# plt.scatter(X[:,0], X[:,1], s=150, linewidths=5)
# plt.show()

colors = 10 * ['g', 'r', 'c', 'b', 'k']

class Mean_Shift:
    def __init__(self, radius=None, radius_norm_step=100):
        # here we have hardcoded radius but it is good over we decided to change and make steps to what radius should be according to data
        self.radius = radius
        self.radius_norm_step =radius_norm_step
        

    def fit(self, data):

        if self.radius == None:
            all_data_centroid = np.average(data, axis=0  )
            all_data_norm = np.linalg.norm(all_data_centroid)
            self.radius = all_data_norm / self.radius_norm_step


        centriods = {}

        for i in range(len(data)):
            #creating same no of centroids
            # later it will be cohorent 
            centriods[i] = data[i]

        weights = [i for i in range(self.radius_norm_step)][::-1]

        while True:
            new_centriods = []
            for i in centriods:
                # num of points within radius / radius
                in_bandwidth = []
                centroid = centriods[i]

                for featureset in data:
                    #if np.linalg.norm(featureset-centroid) < self.radius:
                    #    in_bandwidth.append(featureset)
                    distance = np.linalg.norm(featureset- centroid)
                    if distance == 0:
                        distance = 0.00000000001
                    weight_index = int(distance/self.radius)
                    if weight_index > self.radius_norm_step-1:
                        weight_index = self.radius_norm_step-1

                    to_add = (weights[weight_index])*[featureset]
                    in_bandwidth +=to_add

                # generate new centriod location via mean avg
                new_centriod = np.average(in_bandwidth, axis=0)
                new_centriods.append(tuple(new_centriod))
            
            # this is for the coherent points removal 
            uniques = sorted(list(set(new_centriods)))

            to_pop = []

            for i in uniques:
                for ii in [i for i in uniques]:
                    if i == ii:
                        pass
                    elif np.linalg.norm(np.array(i)-np.array(ii)) <= self.radius:
                        #print(np.array(i), np.array(ii))
                        to_pop.append(ii)
                        break

            for i in to_pop:
                try:
                    uniques.remove(i)
                except:
                    pass

            prev_centroids = dict(centriods)

            # empty the box and add modiifed centrioids
            centriods = {}
            
            for i in range(len(uniques)):
                centriods[i] = np.array(uniques[i])
            
            optimized = True

            for i in centriods: 
                if not np.array_equal(centriods[i], prev_centroids[i]):
                    optimized = False
                if not optimized:
                    break
            
            if optimized:
                break
        
        self.centroids = centriods

        self.classifications = {}

        for i in range(len(self.centroids)):
            self.classifications[i] = []
            
        for featureset in data:
            #compare distance to either centroid
            distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
            #print(distances)
            classification = (distances.index(min(distances)))

            # featureset that belongs to that cluster
            self.classifications[classification].append(featureset)

    def predict(self, data):
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
            #print(distances)
        classification = (distances.index(min(distances)))


clf = Mean_Shift()
clf.fit(X)

centriods = clf.centroids

# plt.scatter(X[:,0], X[:,1], s=150)

for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0],featureset[1], marker='x', color=color, s=150, linewidths=4)

for c in centriods:
    plt.scatter(centriods[c][0], centriods[c][1], color='k', marker='*', s=150)

plt.show()