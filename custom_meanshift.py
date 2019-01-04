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
               [8,2],
               [10,2],
               [9,3],
                     [8,9],
                     [0,3],
                     [5,4],
                     [7,8],
                     [6,4],]))

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
                centriod = centriods[i]

                for featureset in data:
                    # if np.linalg.norm(featureset - centriod) < self.radius:
                    #     in_bandwidth.append(featureset)
                    distance = np.linalg.norm(featureset - centriod)
                    if distance == 0 :
                        distance = 0.00000001
                    
                    weights_index = int(distance/self.radius)
                    if weight_index > self.radius_norm_step-1:
                        weight_index = self.radius_norm_step-1

                    to_add = (weights[weight_index]**2)*[featureset]
                    in_bandwidth +=to_add

                    

                
                # generate new centriod location via mean avg
                new_centriod = np.average(in_bandwidth, axis=0)
                new_centriods.append(tuple(new_centriod))
            
            # this is for the coherent points removal 
            uniques = sorted(list(set(new_centriods)))

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

    def predict(self, data):
        pass

clf = Mean_Shift()
clf.fit(X)

centriods = clf.centroids

plt.scatter(X[:,0], X[:,1], s=150)

for c in centriods:
    plt.scatter(centriods[c][0], centriods[c][1], color='k', marker='*', s=150)

plt.show()