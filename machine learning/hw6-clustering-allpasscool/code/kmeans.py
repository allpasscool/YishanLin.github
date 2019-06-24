import numpy as np

class KMeans():
    def __init__(self, n_clusters):
        """
        This class implements the traditional KMeans algorithm with hard assignments:

        https://en.wikipedia.org/wiki/K-means_clustering

        The KMeans algorithm has two steps:

        1. Update assignments
        2. Update the means

        While you only have to implement the fit and predict functions to pass the
        test cases, we recommend that you use an update_assignments function and an
        update_means function internally for the class.

        Use only numpy to implement this algorithm.

        Args:
            n_clusters (int): Number of clusters to cluster the given data into.

        """
        self.n_clusters = n_clusters
        self.means = None

    def fit(self, features):
        """
        Fit KMeans to the given data using `self.n_clusters` number of clusters.
        Features can have greater than 2 dimensions.

        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            None (saves model - means - internally)
        """
        mean = features[features.shape[0] - self.n_clusters - 1: features.shape[0] - 1, :]
        data_size = features.shape[0]
        labels = np.zeros(data_size)

        while True:
            for i in range(data_size):
                euc_distance = np.sum((mean - features[i]) ** 2, axis = 1)
                labels[i] = np.argmin(euc_distance)

            new_mean = np.zeros((self.n_clusters, features.shape[1]))

            for i in range(self.n_clusters):
                index = np.where(labels == i)
                new_mean[i] = np.mean(features[index], axis = 0)

            done = True

            for i in range(mean.shape[0]):
                over = False
                for j in range(mean.shape[1]):
                    if np.abs(mean[i, j] - new_mean[i, j]) > 10 ** (-5):
                        over = True
                        done = False
                        break
                if over:
                    break
            
            if done:
                break
            else:
                mean = new_mean
            
        self.means = mean

        


    def predict(self, features):
        """
        Given features, an np.ndarray of size (n_samples, n_features), predict cluster
        membership labels.

        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            predictions (np.ndarray): predicted cluster membership for each features,
                of size (n_samples,). Each element of the array is the index of the
                cluster the sample belongs to.
        """
        labels = []

        for i in range(features.shape[0]):
            euc_distance = np.sum((self.means - features[i]) ** 2, axis = 1)
            labels.append(np.argmin(euc_distance))
        
        return np.array(labels)