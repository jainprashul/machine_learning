import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

class Support_Vector_Machine:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1:'r', -1:'b'}
        # for visualization
        if self.visualization :
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)

    #train
    def fit(self, data):
        self.data = data
        # { |w| : [w,b]}
        opt_dict = {}
        
        tranforms = [[1,1],
                    [-1,1],
                    [-1,-1],
                    [1,-1]]
        
        # getting the max and min values from the data
        all_data =[]
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)
        
        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None

        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.05,
                      self.max_feature_value * 0.01,
                      # point of expense :
                      self.max_feature_value * 0.001,]

        # extemely expensive
        b_range_multiple = 5
        #
        b_multiple = 5
        # greatest w initalize curve edges
        latest_optimum = self.max_feature_value*10

        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            # we do this as it is convex
            optimized = False
            while not optimized:
                pass






    
    def predict(self, features):
        # sign(x.w+b)
        classification = np.sign(np.dot(np.array(features), self.w)+ self.b)
        return classification 


 # datasets 
data_dict = {-1:np.array([[1,7],
                          [2.8],
                          [3,8]]),

            1:np.array( [[5,1],
                         [6,-1],
                         [7,3]])
}