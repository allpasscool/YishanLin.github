import numpy as np 
from .distances import euclidean_distances, manhattan_distances, cosine_distances

class KNearestNeighbor():    
    def __init__(self, n_neighbors, distance_measure='euclidean', aggregator='mode'):
        """
        K-Nearest Neighbor is a straightforward algorithm that can be highly
        effective. Training time is...well...is there any training? At test time, labels for
        new points are predicted by comparing them to the nearest neighbors in the
        training data.

        ```distance_measure``` lets you switch between which distance measure you will
        use to compare data points. The behavior is as follows:

        If 'euclidean', use euclidean_distances, if 'manhattan', use manhattan_distances,
        if  'cosine', use cosine_distances.

        ```aggregator``` lets you alter how a label is predicted for a data point based 
        on its neighbors. If it's set to `mean`, it is the mean of the labels of the
        neighbors. If it's set to `mode`, it is the mode of the labels of the neighbors.
        If it is set to median, it is the median of the labels of the neighbors. If the
        number of dimensions returned in the label is more than 1, the aggregator is
        applied to each dimension independently. For example, if the labels of 3 
        closest neighbors are:
            [
                [1, 2, 3], 
                [2, 3, 4], 
                [3, 4, 5]
            ] 
        And the aggregator is 'mean', applied along each dimension, this will return for 
        that point:
            [
                [2, 3, 4]
            ]

        Arguments:
            n_neighbors {int} -- Number of neighbors to use for prediction.
            distance_measure {str} -- Which distance measure to use. Can be one of
                'euclidean', 'manhattan', or 'cosine'. This is the distance measure
                that will be used to compare features to produce labels. 
            aggregator {str} -- How to aggregate a label across the `n_neighbors` nearest
                neighbors. Can be one of 'mode', 'mean', or 'median'.
        """
        self.n_neighbors = n_neighbors
        self.distance_measure = distance_measure
        self.aggregator = aggregator
        self.features = 0
        self.targets = 0
        

    def fit(self, features, targets):
        """Fit features, a numpy array of size (n_samples, n_features). For a KNN, this
        function should store the features and corresponding targets in class 
        variables that can be accessed in the `predict` function. Note that targets can
        be multidimensional! 
        
        HINT: One use case of KNN is for imputation, where the features and the targets 
        are the same. See tests/test_collaborative_filtering for an example of this.
        
        Arguments:
            features {np.ndarray} -- Features of each data point, shape of (n_samples,
                n_features).
            targets {[type]} -- Target labels for each data point, shape of (n_samples, 
                n_dimensions).
        """

        self.features = features
        self.targets = targets
        

    def predict(self, features, ignore_first=False):
        """Predict from features, a numpy array of size (n_samples, n_features) Use the
        training data to predict labels on the test features. For each testing sample, compare it
        to the training samples. Look at the self.n_neighbors closest samples to the 
        test sample by comparing their feature vectors. The label for the test sample
        is the determined by aggregating the K nearest neighbors in the training data.

        Note that when using KNN for imputation, label has shape (1, n_features).

        Arguments:
            features {np.ndarray} -- Features of each data point, shape of (n_samples,
                n_features).
            ignore_first {bool} -- If this is True, then we ignore the closest point
                when doing the aggregation. This is used for collaborative
                filtering, where the closest point is itself and thus is not a neighbor. 
                In this case, we would use 1:(n_neighbors + 1).

        Returns:
            labels {np.ndarray} -- Labels for each data point, of shape (n_samples,
                n_features)
        """
        """
        print("prediction")
        print(self.targets.shape[1])
        print(self.aggregator)

        distance = 0
        
        #get distance
        if self.distance_measure == 'euclidean':
            distance = euclidean_distances(features, self.features)
        elif self.distance_measure == 'manhattan':
            distance = manhattan_distances(features, self.features)
        elif self.distance_measure == 'cosine':
            distance = cosine_distances(features, self.features)

        # get nearest n neighbors
        nearest = []


        for i in range(features.shape[0]):
            tmp = []
            for n in range(self.n_neighbors):
                smallest = 9999999999999999
                smallestIdx = 999999999999999
                for j in range(features.shape[0]):
                    if ignore_first:
                        if distance[i, j] < smallest and (j not in tmp) and (i != j):
                            smallest = distance[i, j]
                            smallestIdx = j
                    else:
                        if distance[i, j] < smallest and (j not in tmp):
                            smallest = distance[i, j]
                            smallestIdx = j
                tmp.append(smallestIdx)
            nearest.append(tmp)
        print(nearest)

        #replace 0s
        #print()
        #print(input_array)
        prediction = []
        for i in range(features.shape[0]):
            #for j in range(features.shape[1]): 
            tmp = []
            for k in nearest[i]:
                #print(input_array[k, j])
                tmp.append(self.targets[k])
            #print(nearest[i])
            #print(tmp)
            #print(self.targets.shape[1])
            if self.targets.shape[1] > 1:
                tmp = np.array(tmp)
            if self.targets.shape[1] == 1:
                tmp = np.array(tmp)
            else:
                print(tmp)
            t = []
            if self.aggregator == 'mode':
                print(tmp)
                tmp1 = []
                if self.targets.shape[1] > 1:
                    for m in range(tmp.shape[1]):
                        tmp2 = []
                        for h in range(tmp.shape[0]):
                            tmp2.append(self.targets[h, m])
                        g = max(set(tmp2), key=tmp2.count)
                        tmp1.append(g)
                else:
                    tmp2 = []
                    for h in range(len(tmp)):
                        tmp2.append(self.targets[h])
                    g = max(set(tmp2), key=tmp2.count)
                    tmp1 = g
                t.append(tmp1)
                #print(t)
            elif self.aggregator == 'mean':
                tmp1 = []
                if self.targets.shape[1] > 1:
                    for m in range(tmp.shape[1]):
                        sumN = 0
                        for h in range(tmp.shape[0]):
                            sumN += self.targets[h, m]
                        g = sumN / tmp.shape[0]
                        tmp1.append(g)
                else:
                    sumN = 0
                    for h in range(len(tmp)):
                        sumN += self.targets[h]
                    g = sumN / len(tmp)
                    tmp1 = g
                t.append(tmp1)
            elif self.aggregator == 'median':
                tmp1 = []
                if self.targets.shape[1] > 1:
                    for m in range(tmp.shape[1]):
                        tmp2 = []
                        for h in range(tmp.shape[0]):
                            tmp2.append(self.targets[h, m])
                        tmp2.sort()
                        g = 0
                        if len(tmp2) % 2 == 0:
                            g = (tmp2[len(tmp2)/2] + tmp2[len(tmp2)/2-1]) /2
                        else:
                            g = tmp2[int(len(tmp2)/2)]
                        tmp1.append(g)
                else:
                    tmp2 = []
                    for h in range(len(tmp)):
                        tmp2.append(self.targets[h])
                    tmp2.sort()
                    g = 0
                    if len(tmp2) % 2 == 0:
                        g = (tmp2[len(tmp2)/2] + tmp2[len(tmp2)/2-1]) /2
                    else:
                        g = tmp2[int(len(tmp2)/2)]
                    tmp1 = g
                t.append(tmp1)
            prediction.append(t)
        
        print(prediction)
        if self.targets.shape[1] > 1:
            return prediction
        return np.array(prediction)
        """
        
        distance = 0

        #get distance
        if self.distance_measure == 'euclidean':
            distance = euclidean_distances(features, self.features)
        elif self.distance_measure == 'manhattan':
            distance = manhattan_distances(features, self.features)
        elif self.distance_measure == 'cosine':
            distance = cosine_distances(features, self.features)

        #nearest
        prediction = []
        for i in distance:
            indexes = np.argsort(i)
            if not ignore_first:
                indexes = indexes[:self.n_neighbors]
            elif ignore_first:
                indexes = indexes[1:self.n_neighbors+1,]
            train_targets = self.targets[indexes,:]  
            tmp = []

            
            if self.aggregator == 'mean':
                for j in range(train_targets.shape[1]):
                    tmp1 = train_targets[:,j]
                    tmp.append(np.mean(tmp1))
            elif self.aggregator == 'mode':
                for j in range(train_targets.shape[1]):
                    tmp1 = train_targets[:,j]
                    tmp2 =[]
                    for k in range(np.size(tmp1)):
                        tmp2.append(tmp1[k])
                    tmp3 = np.unique(tmp2)

                    frequency = [0] * len(tmp3)
                    for k in tmp2:
                        for h in range(len(tmp3)):
                            if k == tmp3[h]:
                                frequency[h] += 1
                                break
                    most_frequency = frequency.index(max(frequency))
                    tmp.append(tmp3[most_frequency] * 1.0)
            elif self.aggregator == 'median':
                for j in range(train_targets.shape[1]):
                    tmp1 = train_targets[:,j]
                    tmp.append(np.median(tmp1))
            prediction.append(tmp)
        
        return prediction