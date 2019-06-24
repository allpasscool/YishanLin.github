import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

def transform_data(features):
    """
    Data can be transformed before being put into a linear discriminator. If the data
    is not linearly separable, it can be transformed to a space where the data
    is linearly separable, allowing the perceptron algorithm to work on it. This
    function should implement such a transformation on a specific dataset (NOT 
    in general).

    Args:
        features (np.ndarray): input features
    Returns:
        transformed_features (np.ndarray): features after being transformed by the function
    """
    #print("transform_data")
    #print(features)
    
    tran_features = np.zeros([features.shape[0], features.shape[1]], dtype=float)
    for i in range(features.shape[0]):
        #tran_features[i, 0] = (abs(features[i, 0] + 50) - abs(features[i, 1])) ** 2
        #tran_features[i, 1] = (abs(features[i, 0] + 50) - abs(features[i, 1])) ** 2
        tran_features[i, 0] = abs(features[i, 0])
        tran_features[i, 1] = abs(features[i, 1])        
        #tran_features[i, 0] = (features[i, 0] + features[i, 1])
        #tran_features[i, 1] = (features[i, 0] + features[i, 1])
        #tran_features[i, 0] = (features[i, 0] * features[i, 0]) + (features[i, 1] * features[i, 1])
        #tran_features[i, 1] = (features[i, 0] * features[i, 0]) + (features[i, 1] * features[i, 1])
    #print("after transform")
    #print(features)

    return tran_features
    
class Perceptron():
    def __init__(self, max_iterations=200):
        """
        This implements a linear perceptron for classification. A single
        layer perceptron is an algorithm for supervised learning of a binary
        classifier. The idea is to draw a linear line in the space that separates
        the points in the space into two partitions. Points on one side of the 
        line are one class and points on the other side are the other class.
       
        begin initialize weights
            while not converged or not exceeded max_iterations
                for each example in features
                    if example is misclassified using weights
                    then weights = weights + example * label_for_example
            return weights
        end
        
        Note that label_for_example is either -1 or 1.

        Use only numpy to implement this algorithm. 

        Args:
            max_iterations (int): the perceptron learning algorithm stops after 
            this many iterations if it has not converged.

        """
        self.max_iterations = max_iterations
        self.w = 0

    def fit(self, features, targets):
        """
        Fit a single layer perceptron to features to classify the targets, which
        are classes (-1 or 1). This function should terminate either after
        convergence (dividing line does not change between interations) or after
        max_iterations (defaults to 200) iterations are done. Here is pseudocode for 
        the perceptron learning algorithm:

        Args:
            features (np.ndarray): 2D array containing inputs.
            targets (np.ndarray): 1D array containing binary targets.
        Returns:
            None (saves model and training data internally)
        """
        
        #m = size of data
        m = features.shape[0]
        w = np.random.uniform(0, 0, size=(features.shape[1] + 1))
        k = 0

        for i in range(self.max_iterations):
            for t in range(features.shape[0] * 10):
                converge = True
                k = (k+1) % m
                x = np.append([1], features[k])
                if self.h(w, x) != targets[k]:
                    #w += x * (targets[k] - self.h(w, x))
                    w += x * targets[k]
                    converge = False
                    continue
                else:
                    for j in range(m):
                        if(self.h(w, x) != targets[(k + j) % m]):
                            converge = False
                            break
                    if converge:
                        self.w = w
                        return
        self.w = w

    def h(self, W, x):
        if np.dot(W, x) > 0:
            return 1
        else:
            return -1

    def predict(self, features):
        """
        Given features, a 2D numpy array, use the trained model to predict target 
        classes. Call this after calling fit.

        Args:
            features (np.ndarray): 2D array containing real-valued inputs.
        Returns:
            predictions (np.ndarray): Output of saved model on features.
        """
        
        ans = np.random.uniform(0, 0, size=(features.shape[0]))
        
        for i in range(features.shape[0]):
            ans[i] = self.h(self.w, np.append([1], features[i]))
        
        return ans

    def visualize(self, features, targets):
        """
        This function should produce a single plot containing a scatter plot of the
        features and the targets, and the perceptron fit by the model should be
        graphed on top of the points.

        DO NOT USE plt.show() IN THIS FUNCTION.

        Args:
            features (np.ndarray): 2D array containing real-valued inputs.
            targets (np.ndarray): 1D array containing binary targets.
        Returns:
            None (plots to the active figure)
        """

        plt.figure(figsize=(6,4))
        plt.scatter(features[:, 0], features[:, 1], c=targets)
        plt.title("perceptron")
        plt.savefig("perceptron.png")
        #raise NotImplementedError()