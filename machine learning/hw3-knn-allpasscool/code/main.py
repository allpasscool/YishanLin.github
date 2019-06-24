from load_movielens import load_movielens_data
import numpy as np
import random
try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt


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
def euclidean_distances(X, Y):
    """Compute pairwise Euclidean distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Euclidean distance between two rows.
    
    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Euclidean distances between rows of X and rows of Y.
    """
    print("enter euclidean_distances")
    M = X.shape[0]
    N = Y.shape[0]
    K = X.shape[1]
    D = np.zeros([M, N])
    for i in range(0, M):
        print(i)
        for j in range(0, N):
            tmp = 0
            for k in range(0, K):
                tmp += (X[i, k] - Y[j, k]) ** 2
            D[i, j] = np.sqrt(tmp)
    return D

def manhattan_distances(X, Y):
    """Compute pairwise Manhattan distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Manhattan distance between two rows.
    
    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Manhattan distances between rows of X and rows of Y.
    """
    print("manhattan")
    M = X.shape[0]
    N = Y.shape[0]
    K = X.shape[1]
    D = np.zeros([M, N])

    for i in range(0, M):
        print(i)
        for j in range(0, N):
            tmp = 0
            for k in range(0, K):
                tmp += abs(X[i, k] - Y[j, k])
            D[i, j] = tmp
    return D


def cosine_distances(X, Y):
    """Compute Cosine distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Cosine distance between two rows.
    
    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Cosine distances between rows of X and rows of Y.
    """
    print("enter cosine")
    M = X.shape[0]
    N = Y.shape[0]
    K = X.shape[1]
    D = np.zeros([M, N])

    for i in range(0, M):
        print(i)
        for j in range(0, N):
            tmp = 0
            tmp1 = 0
            tmp2 = 0
            for k in range(0, K):
                tmp += (X[i, k] * Y[j, k])
                tmp1 += X[i, k] * X[i, k]
                tmp2 += Y[j, k] * Y[j, k]
                
            tmp1 = np.sqrt(tmp1)
            tmp2 = np.sqrt(tmp2)
            D[i, j] = 1 - (tmp / (tmp1 * tmp2))
            #D[i, j] = tmp / tmp1
    return D


def collaborative_filtering(input_array, n_neighbors,
                            distance_measure='euclidean', aggregator='mode'):
    """
    This is a wrapper function for your KNearestNeighbors class, that runs kNN
    as a collaborative filter.

    If there is a 0 in the array, you must impute a value determined by using your
    kNN classifier as a collaborative filter. All non-zero entries should remain
    the same.

    For example, if `input_array`(containing data we are trying to impute) looks like:

        [[0, 2],
         [1, 2],
         [1, 0]]

    We are trying to impute the 0 values by replacing the 0 values with an aggregation of the
    neighbors for that row. The features that are 0 in the row are replaced with an aggregation
    of the corresponding column of the neighbors of that row. For example, if aggregation is 'mean', 
    and K = 2 then the output should be:

        [[1, 2],
         [1, 2],
         [1, 2]]

    Note that the row you are trying to impute for is ignored in the aggregation. 
    Use `ignore_first = True` in the predict function of the KNN to accomplish this. If 
    `ignore_first = False` and K = 2, then the result would be:

        [[(1 + 0) / 2 = .5, 2],
         [1, 2],
         [1, (2 + 0) / 2 = 1]]

        = [[.5, 2],
           [1, 2],
           [1, 1]]

    This is incorrect because the value that we are trying to replace is considered in the
    aggregation.

    The non-zero values are left untouched. If aggregation is 'mode', then the output should be:

        [[1, 2],
         [1, 2],
         [1, 2]]


    Arguments:
        input_array {np.ndarray} -- An input array of shape (n_samples, n_features).
            Any zeros will get imputed.
        n_neighbors {int} -- Number of neighbors to use for prediction.
        distance_measure {str} -- Which distance measure to use. Can be one of
            'euclidean', 'manhattan', or 'cosine'. This is the distance measure
            that will be used to compare features to produce labels.
        aggregator {str} -- How to aggregate a label across the `n_neighbors` nearest
            neighbors. Can be one of 'mode', 'mean', or 'median'.

    Returns:
        imputed_array {np.ndarray} -- An array of shape (n_samples, n_features) with imputed
            values for any zeros in the original input_array.
    """
    print("enter collaborative_filtering")
    distance = 0
    
    #get distance
    if distance_measure == 'euclidean':
        distance = euclidean_distances(input_array, input_array)
    elif distance_measure == 'manhattan':
        distance = manhattan_distances(input_array, input_array)
    elif distance_measure == 'cosine':
        distance = cosine_distances(input_array, input_array)

    # get nearest n neighbors
    nearest = []


    for i in range(input_array.shape[0]):
        tmp = []
        for n in range(n_neighbors):
            smallest = 9999999999999999
            smallestIdx = 999999999999999
            for j in range(input_array.shape[0]):
                if distance[i, j] < smallest and (j not in tmp) and (i != j):
                    smallest = distance[i, j]
                    smallestIdx = j
            tmp.append(smallestIdx)
        nearest.append(tmp)

    #replace 0s
    #print()
    #print(input_array)
    for i in range(input_array.shape[0]):
        for j in range(input_array.shape[1]): 
            tmp = []
            if input_array[i,j] == 0:
                for k in nearest[i]:
                    #print(input_array[k, j])
                    tmp.append(input_array[k, j])
                #print(nearest[i])
                #print(tmp)
                t = 0
                if aggregator == 'mode':
                    t = max(set(tmp), key=tmp.count)
                    #print(t)
                elif aggregator == 'mean':
                    sumN = 0
                    for k in tmp:
                        sumN += k
                    t = sumN / len(tmp)

                elif aggregator == 'median':
                    tmp.sort()
                    if len(tmp) % 2 == 0:
                        t = (tmp[len(tmp)/2] + tmp[len(tmp)/2-1]) /2
                    else:
                        t = tmp[int(len(tmp)/2)]
                input_array[i, j] = t

    #print(input_array)
    return input_array


def free_res(N, K, D, A, rating, newOne, inD):
    print("enter free res")
    inputD = np.empty_like(rating)
    inputD[:] = rating    
    if newOne == False:
        inputD = inD

    #N
    if newOne == True:
        for i in range(943):
            count = 0
            GG = 0
            while True:
                j = random.randint(0, 1681)
                GG += 1
                if inputD[i, j] != 0 and inputD[i,j] != 0:
                    inputD[i, j] = 0
                    count += 1
                    if count == N:
                        break
                if(GG == 1000):
                    print("GG")
                    break
        
    #
    if newOne == True:
        inputD = collaborative_filtering(inputD, K, D, A)


    #MSE
    MSE = 0
    counter = 0
    for i in range(943):
        for j in range(1682):
            if rating[i,j] != 0:
                MSE += (rating[i,j] - inputD[i,j]) ** 2
                counter += 1
    MSE /= counter
    MSE = np.sqrt(MSE)
    print("MSE")
    print(MSE)
    return MSE, inputD


def main():
    """
    rating = load_movielens_data("..\data\ml-100k")
    #print(rating)
    
    common = []
    for i in range(943):
        for j in range(i, 943):
            count = 0
            for k in range(1682):
                if  rating[i,k] > 0 and rating[i, k] == rating[j, k]:
                    count += 1
            common.append(count)

    common.sort()

    tmp = 0
    counter = 0
    for i in common:
        tmp += i
        counter += 1
    mean = tmp / (counter)
    print("mean")
    print(mean)
    print("len common")
    print(len(common))
    print("len common /2")
    print(len(common)/2)

    meidan = 0
    if counter % 2 == 0:
        a = int(counter/2 -1)
        b = int(counter/2)
        median = (common[a] + common[b]) /2
    else:
        a = int(counter/2)
        median = common[a]
    print("median")
    print(median)
    """
    rating = load_movielens_data("..\data\ml-100k")
    #8
    #MSE1 = free_res(1, 3, 'euclidean', 'mean', rating)
    #MSE2 = free_res(2, 3, 'euclidean', 'mean', rating)
    #MSE4 = free_res(4, 3, 'euclidean', 'mean', rating)
    #MSE8 = free_res(8, 3, 'euclidean', 'mean', rating)

    #9
    #MSE_1 = free_res(1, 3, 'euclidean', 'mean', rating)
    #MSE_2 = free_res(1, 3, 'cosine', 'mean', rating)
    #MSE_3 = free_res(1, 3, 'manhattan', 'mean', rating)
    
    #10
    #MSE1, aa = free_res(1, 1, 'cosine', 'mean', rating, True, 0)
    #MSE3, aa = free_res(1, 3, 'cosine', 'mean', rating, False, aa)
    #MSE7, aa = free_res(1, 7, 'cosine', 'mean', rating, False, aa)
    #MSE11, aa = free_res(1, 11, 'cosine', 'mean', rating, False, aa)
    #MSE15, aa = free_res(1, 15, 'cosine', 'mean', rating, False, aa)
    #MSE31, aa = free_res(1, 31, 'cosine', 'mean', rating, False, aa)

    #11
    MSE_1, aa = free_res(1, 7, 'cosine', 'mean', rating, True, 0)
    MSE_2, aa = free_res(1, 7, 'cosine', 'mode', rating, False, aa)
    MSE_3, aa = free_res(1, 7, 'cosine', 'median', rating, False, aa)
    MSE_4, aa = free_res(1, 11, 'cosine', 'mean', rating, True, 0)
    MSE_5, aa = free_res(1, 11, 'cosine', 'mode', rating, False, aa)
    MSE_6, aa = free_res(1, 11, 'cosine', 'median', rating, False, aa)
    MSE_7, aa = free_res(1, 15, 'cosine', 'mean', rating, True, 0)
    MSE_8, aa = free_res(1, 15, 'cosine', 'mode', rating, False, aa)
    MSE_9, aa = free_res(1, 15, 'cosine', 'median', rating, False, aa)
    MSE_10, aa = free_res(1, 31, 'cosine', 'mean', rating, True, 0)
    MSE_11, aa = free_res(1, 31, 'cosine', 'mode', rating, False, aa)
    MSE_12, aa = free_res(1, 31, 'cosine', 'median', rating, False, aa)
    
    tmp = []
    #x = np.random.uniform(-20, 20, size=(4))
    #y = np.random.uniform(-20, 20, size=(4))
    #x[0] = 1
    #x[1] = 2
    #x[2] = 4
    #x[3] = 8
    #y[0] = MSE1
    #y[1] = MSE2
    #y[2] = MSE4
    #y[3] = MSE8
    
    
    x = np.random.uniform(-20, 20, size=(12))
    y = np.random.uniform(-20, 20, size=(12))
    x[0] = 1
    x[1] = 2
    x[2] = 3
    x[3] = 4
    x[4] = 5 
    x[5] = 6
    x[6] = 7
    x[7] = 8
    x[8] = 9
    x[9] = 10
    x[10] = 11
    x[11] = 12
    y[0] = MSE_1
    y[1] = MSE_2
    y[2] = MSE_3
    y[3] = MSE_4
    y[4] = MSE_5
    y[5] = MSE_6
    y[6] = MSE_7
    y[7] = MSE_8
    y[8] = MSE_9
    y[9] = MSE_10
    y[10] = MSE_11
    y[11] = MSE_12


    
    plt.figure(figsize=(6,4))
    plt.xlabel('K', fontsize=10)
    plt.ylabel('mean_squared_error', fontsize=10)
    plt.scatter(x, y)
    #plt.plot(, , label ='np.log10 test MSE')
    #plt.scatter(features[:, 0], features[:, 1], label='pred')
    plt.legend()
    #plt.savefig('question8' + ".png")
    plt.savefig('question11' + ".png")
    #plt.savefig('question10' + ".png")
    #plt.savefig('question11' + ".png")
    
if __name__ == "__main__":
    main()