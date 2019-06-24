import numpy as np
# from your_code import GradientDescent
# from your_code import HingeLoss, SquaredLoss
# from your_code import L1Regularization, L2Regularization
import json
import os
import struct
from array import array as pyarray
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random


class Regularization:
    """
    Abstract base class for regularization terms in gradient descent.

    *** THIS IS A BASE CLASS: YOU DO NOT NEED TO IMPLEMENT THIS ***

    Arguments:
        reg_param - (float) The hyperparameter that controls the amount of
            regularization to perform. Must be non-negative.
    """

    def __init__(self, reg_param=0.05):
        self.reg_param = reg_param

    def forward(self, w):
        """
        Implements the forward pass through the regularization term.

        *** THIS IS A BASE CLASS: YOU DO NOT NEED TO IMPLEMENT THIS ***

        Arguments:
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
        Returns:
            regularization_term - (float) The value of the regularization term
                evaluated at w.
        """
        pass

    def backward(self, w):
        """
        Implements the backward pass through the regularization term.

        *** THIS IS A BASE CLASS: YOU DO NOT NEED TO IMPLEMENT THIS ***

        Arguments:
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
        Returns:
            gradient_term - (np.array) A numpy array of length d+1. The
                gradient of the regularization term evaluated at w.
        """
        pass


class L1Regularization(Regularization):
    """
    L1 Regularization for gradient descent.
    """

    def forward(self, w):
        """
        Implements the forward pass through the regularization term. For L1,
        this is the L1-norm of the model parameters weighted by the
        regularization parameter. Note that the bias (the last value in w)
        should NOT be included in regularization.

        Arguments:
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
        Returns:
            regularization_term - (float) The value of the regularization term
                evaluated at w.
        """
        return np.sum(np.abs(w[:-1])) * self.reg_param

    def backward(self, w):
        """
        Implements the backward pass through the regularization term. The
        backward pass is the gradient of the forward pass with respect to the
        model parameters.

        Arguments:
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
        Returns:
            gradient_term - (np.array) A numpy array of length d+1. The
                gradient of the regularization term evaluated at w.
        """
        w[-1] = 0

        return np.sign(w) * self.reg_param


class L2Regularization(Regularization):
    """
    L2 Regularization for gradient descent.
    """

    def forward(self, w):
        """
        Implements the forward pass through the regularization term. For L2,
        this is half the squared L2-norm of the model parameters weighted by
        the regularization parameter. Note that the bias (the last value in w)
        should NOT be included in regularization.

        Arguments:
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
        Returns:
            regularization_term - (float) The value of the regularization term
                evaluated at w.
        """
        
        return (1/2) * np.sum(np.abs(w[:-1]) ** 2) * self.reg_param

    def backward(self, w):
        """
        Implements the backward pass through the regularization term. The
        backward pass is the gradient of the forward pass with respect to the
        model parameters.

        Arguments:
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
        Returns:
            gradient_term - (np.array) A numpy array of length d+1. The
                gradient of the regularization term evaluated at w.
        """
        w[-1] = 0

        return w * self.reg_param

def accuracy(ground_truth, predictions):
    """
    Reports the classification accuracy.

    Arguments:
        ground_truth - (np.array) A 1D numpy array of length N. The true class
            labels.
        predictions - (np.array) A 1D numpy array of length N. The class labels
            predicted by the model.
    Returns:
        accuracy - (float) The accuracy of the predictions.
    """
    return np.mean(ground_truth == predictions)


def confusion_matrix(ground_truth, predictions):
    """
    Reports the classification accuracy.

    Arguments:
        ground_truth - (np.array) A 1D numpy array of length N. The true class
            labels.
        predictions - (np.array) A 1D numpy array of length N. The class labels
            predicted by the model.
    Returns:
        confusion_matrix - (np.array) The confusion matrix. A CxC numpy array,
            where C is the number of unique classes. Index i, j is the number
            of times an example belonging to class i was predicted to belong
            to class j.
    """
    classes = np.unique(ground_truth)
    confusion = np.zeros((len(classes), len(classes)))
    for i, prediction in enumerate(predictions):
        confusion[ground_truth[i], prediction] += 1
    return confusion


class MultiClassGradientDescent:
    """
    Implements linear gradient descent for multiclass classification. Uses
    One-vs-All (OVA) classification for aggregating binary classification
    results to the multiclass setting.

    Arguments:
        loss - (string) The loss function to use. One of 'hinge' or 'squared'.
        regularization - (string or None) The type of regularization to use.
            One of 'l1', 'l2', or None. See regularization.py for more details.
        learning_rate - (float) The size of each gradient descent update step.
        reg_param - (float) The hyperparameter that controls the amount of
            regularization to perform. Must be non-negative.
    """

    def __init__(self, loss, regularization=None,
                 learning_rate=0.01, reg_param=0.05):
        self.loss = loss
        self.regularization = regularization
        self.learning_rate = learning_rate
        self.reg_param = reg_param

        self.model = []
        self.classes = None

    def fit(self, features, targets, batch_size=None, max_iter=1000):
        """
        Fits a multiclass gradient descent learner to the features and targets
        by using One-vs-All classification. In other words, for each of the c
        output classes, train a GradientDescent classifier to determine whether
        each example does or does not belong to that class.

        Store your c GradientDescent classifiers in the list self.model. Index
        c of self.model should correspond to the binary classifier trained to
        predict whether examples do or do not belong to class c.

        Arguments:
            features - (np.array) An Nxd array of features, where N is the
                number of examples and d is the number of features.
            targets - (np.array) A 1D array of targets of size N. Contains c
                unique values (the possible class labels).
            batch_size - (int or None) The number of examples used in each
                iteration. If None, use all of the examples in each update.
            max_iter - (int) The maximum number of updates to perform.
        Modifies:
            self.model - (list) A list of c GradientDescent objects. The models
                trained to perform OVA classification for each class.
            self.classes - (np.array) A numpy array of the unique target
                values. Required to associate a model index with a target value
                in predict.
        """
        """
        classes = np.unique(targets)
        model = []

        for i in classes:
            tmp = targets.copy()
            tmp[np.where(tmp != i)] = -10
            tmp[np.where(tmp == i)] = -8
            
            tmp_model = GradientDescent(self.loss, self.regularization, self.learning_rate, self.reg_param)
            tmp_model.fit(features, tmp, max_iter=max_iter)

            model.append(tmp_model)

        self.classes = classes
        self.model = model
        """
        self.classes = np.unique(targets)
        model = []
        for c in self.classes:
            temp = targets.copy()
            temp[np.where(temp != c)] = -10
            temp[np.where(temp == c)] = -8
            learner = GradientDescent(self.loss,self.regularization,self.learning_rate,self.reg_param)
            learner.fit(features,temp,max_iter)
            model.append(learner)
        self.model = model


    def predict(self, features):
        """
        Predicts the class labels of each example in features using OVA
        aggregation. In other words, predict as the output class the class that
        receives the highest confidence score from your c GradientDescent
        classifiers. Predictions should be in the form of integers that
        correspond to the index of the predicted class.

        Arguments:
            features - (np.array) A Nxd array of features, where N is the
                number of examples and d is the number of features.
        Returns:
            predictions - (np.array) A 1D array of predictions of length N,
                where index d corresponds to the prediction of row N of
                features.
        """
        """
        prediction = self.model[0].confidence(features)

        for i in range(np.size(self.classes)):
            if i == 0:
                continue
            prediction = np.c_[prediction, self.model[i].confidence(features)]

        prediction = np.argmax(prediction, axis=1)

        for i in range(np.size(prediction)):
            prediction[i] = self.classes[prediction[i]]
        
        return prediction
        """
        matrix = None
        for i in range(np.size(self.classes)):
            if i == 0:
                matrix = self.model[i].confidence(features)
            else:
                matrix = np.c_[matrix,self.model[i].confidence(features)]
        pre = np.argmax(matrix,axis=1)
        for i in range(np.size(pre)):
            pre[i] = self.classes[pre[i]]
        return pre



class Loss:
    """
    An abstract base class for a loss function that computes both the prescribed
    loss function (the forward pass) as well as its gradient (the backward
    pass).

    *** THIS IS A BASE CLASS: YOU DO NOT NEED TO IMPLEMENT THIS ***

    Arguments:
        regularization - (`Regularization` or None) The type of regularization to
            perform. Either a derived class of `Regularization` or None. If None,
            no regularization is performed.
    """

    def __init__(self, regularization=None):
        self.regularization = regularization

    def forward(self, X, w, y):
        """
        Computes the forward pass through the loss function. If
        self.regularization is not None, also adds the forward pass of the
        regularization term to the loss.

        *** THIS IS A BASE CLASS: YOU DO NOT NEED TO IMPLEMENT THIS ***

        Arguments:
            X - (np.array) An Nx(d+1) array of features, where N is the number
                of examples and d is the number of features. The +1 refers to
                the bias term.
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
            y - (np.array) A 1D array of targets of length N.
        Returns:
            loss - (float) The calculated loss normalized by the number of
                examples, N.
        """
        pass

    def backward(self, X, w, y):
        """
        Computes the gradient of the loss function with respect to the model
        parameters. If self.regularization is not None, also adds the backward
        pass of the regularization term to the loss.

        *** THIS IS A BASE CLASS: YOU DO NOT NEED TO IMPLEMENT THIS ***

        Arguments:
            X - (np.array) An Nx(d+1) array of features, where N is the number
                of examples and d is the number of features. The +1 refers to
                the bias term.
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
            y - (np.array) A 1D array of targets of length N.
        Returns:
            gradient - (np.array) The (d+1)-dimensional gradient of the loss
                function with respect to the model parameters. The +1 refers to
                the bias term.
        """
        pass


class SquaredLoss(Loss):
    """
    The squared loss function.
    """

    def forward(self, X, w, y):
        """
        Computes the forward pass through the loss function. If
        self.regularization is not None, also adds the forward pass of the
        regularization term to the loss. The squared loss for a single example
        is given as follows:

        L_s(x, y; w) = (1/2) (y - w^T x)^2

        The squared loss over a dataset of N points is the average of this
        expression over all N examples.

        Arguments:
            X - (np.array) An Nx(d+1) array of features, where N is the number
                of examples and d is the number of features. The +1 refers to
                the bias term.
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
            y - (np.array) A 1D array of targets of length N.
        Returns:
            loss - (float) The calculated loss normalized by the number of
                examples, N.
        """
        
        loss = (1/2) * np.sum((y - np.dot(X, w)) ** 2) / X.shape[0]

        return loss if self.regularization == None else (loss + self.regularization.forward(w))
        """
        loss = 0.5 * np.sum((y - X.dot(w)) ** 2) / X.shape[0]
        if self.regularization == None:
            return loss
        else:
            return loss + self.regularization.forward(w)
        """

    def backward(self, X, w, y):
        """
        Computes the gradient of the loss function with respect to the model
        parameters. If self.regularization is not None, also adds the backward
        pass of the regularization term to the loss.

        Arguments:
            X - (np.array) An Nx(d+1) array of features, where N is the number
                of examples and d is the number of features. The +1 refers to
                the bias term.
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
            y - (np.array) A 1D array of targets of length N.
        Returns:
            gradient - (np.array) The (d+1)-dimensional gradient of the loss
                function with respect to the model parameters. The +1 refers to
                the bias term.
        """
        
        sum = 0
        for i in range(X.shape[0]):
            sum += (y[i] - np.dot(X[i], w)) * -X[i]

        return (sum / X.shape[0]) if self.regularization == None else ((sum / X.shape[0]) + self.regularization.backward(w))
        
        """
        temp = y - X.dot(w)
        de = -X
        for i in range(X.shape[0]):
            de[i] *= temp[i]
        de = np.sum(de,axis=0) / X.shape[0]
        if self.regularization == None:
            return de
        else:
            return de + self.regularization.backward(w)
        """

class HingeLoss(Loss):
    """
    The hinge loss function.

    https://en.wikipedia.org/wiki/Hinge_loss
    """

    def forward(self, X, w, y):
        """
        Computes the forward pass through the loss function. If
        self.regularization is not None, also adds the forward pass of the
        regularization term to the loss. The hinge loss for a single example
        is given as follows:

        L_h(x, y; w) = max(0, 1 - y w^T x)

        The hinge loss over a dataset of N points is the average of this
        expression over all N examples.

        Arguments:
            X - (np.array) An Nx(d+1) array of features, where N is the number
                of examples and d is the number of features. The +1 refers to
                the bias term.
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
            y - (np.array) A 1D array of targets of length N.
        Returns:
            loss - (float) The calculated loss normalized by the number of
                examples, N.
        """
        loss = 0.0

        tmp = 1 - np.dot(X, w) * y

        for i in range(X.shape[0]):
            loss += 0 if tmp[i] < 0 else tmp[i]
        
        return (loss / X.shape[0]) if self.regularization == None else ((loss / X.shape[0]) + self.regularization.forward(w))
        


    def backward(self, X, w, y):
        """
        Computes the gradient of the loss function with respect to the model
        parameters. If self.regularization is not None, also adds the backward
        pass of the regularization term to the loss.

        Arguments:
            X - (np.array) An Nx(d+1) array of features, where N is the number
                of examples and d is the number of features. The +1 refers to
                the bias term.
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
            y - (np.array) A 1D array of targets of length N.
        Returns:
            gradient - (np.array) The (d+1)-dimensional gradient of the loss
                function with respect to the model parameters. The +1 refers to
                the bias term.
        """
        
        gradient = 0.0

        tmp = 1 - np.dot(X, w) * y

        for i in range(X.shape[0]):
            gradient += 0 if tmp[i] <= 0 else -y[i] * X[i]
        
        return (gradient / X.shape[0]) if self.regularization == None else ((gradient / X.shape[0]) + self.regularization.forward(w))



class ZeroOneLoss(Loss):
    """
    The 0-1 loss function.

    The loss is 0 iff w^T x == y, else the loss is 1.

    *** YOU DO NOT NEED TO IMPLEMENT THIS ***
    """

    def forward(self, X, w, y):
        """
        Computes the forward pass through the loss function. If
        self.regularization is not None, also adds the forward pass of the
        regularization term to the loss. The squared loss for a single example
        is given as follows:

        L_0-1(x, y; w) = {0 iff w^T x == y, else 1}

        The squared loss over a dataset of N points is the average of this
        expression over all N examples.

        Arguments:
            X - (np.array) An Nx(d+1) array of features, where N is the number
                of examples and d is the number of features. The +1 refers to
                the bias term.
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
            y - (np.array) A 1D array of targets of length N.
        Returns:
            loss - (float) The average loss.
        """
        predictions = (X @ w > 0.0).astype(int) * 2 - 1
        loss = np.sum((predictions != y).astype(float)) / len(X)
        if self.regularization:
            loss += self.regularization.forward(w)
        return loss

    def backward(self, X, w, y):
        """
        Computes the gradient of the loss function with respect to the model
        parameters. If self.regularization is not None, also adds the backward
        pass of the regularization term to the loss.

        Arguments:
            X - (np.array) An Nx(d+1) array of features, where N is the number
                of examples and d is the number of features. The +1 refers to
                the bias term.
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
            y - (np.array) A 1D array of targets of length N.
        Returns:
            gradient - (np.array) The (d+1)-dimensional gradient of the loss
                function with respect to the model parameters. The +1 refers to
                the bias term.
        """
        # This function purposefully left blank
        raise ValueError('No need to use this function for the homework :p')






def load_data(dataset, fraction=1.0, base_folder='data'):
    """
    Loads a dataset and performs a random stratified split into training and
    test partitions.

    Arguments:
        dataset - (string) The name of the dataset to load. One of the
            following:
              'blobs': A linearly separable binary classification problem.
              'mnist-binary': A subset of the MNIST dataset containing only
                  0s and 1s.
              'mnist-multiclass': A subset of the MNIST dataset containing the
                  numbers 0 through 4, inclusive.
              'synthetic': A small custom dataset for exploring properties of
                  gradient descent algorithms.
        fraction - (float) Value between 0.0 and 1.0 representing the fraction
            of data to include in the training set. The remaining data is
            included in the test set. Unused if dataset == 'synthetic'.
        base_folder - (string) absolute path to your 'data' directory. If
            defaults to 'data'.
    Returns:
        train_features - (np.array) An Nxd array of features, where N is the
            number of examples and d is the number of features.
        test_features - (np.array) An Nxd array of features, where M is the
            number of examples and d is the number of features.
        train_targets - (np.array) A 1D array of targets of size N.
        test_targets - (np.array) A 1D array of targets of size M.
    """
    if dataset == 'blobs':
        path = os.path.join(base_folder, 'blobs.json')
        train_features, test_features, train_targets, test_targets = \
            load_json_data(path)
    elif dataset == 'mnist-binary':
        train_features, test_features, train_targets, test_targets = \
            load_mnist_data(2, fraction=fraction, mnist_folder=base_folder)
        train_targets = train_targets * 2 - 1
        test_targets = test_targets * 2 - 1
    elif dataset == 'mnist-multiclass':
        train_features, test_features, train_targets, test_targets = \
            load_mnist_data(5, fraction=fraction, examples_per_class=100,
                            mnist_folder=base_folder)
    elif dataset == 'synthetic':
        path = os.path.join(base_folder,  'synthetic.json')
        train_features, test_features, train_targets, test_targets = \
            load_json_data(path)
    else:
        raise ValueError('Dataset {} not found!'.format(dataset))

    # Normalize the data using feature-independent whitening. Note that the
    # statistics are computed with respect to the training set and applied to
    # both the training and testing sets.
    if dataset != 'synthetic':
        mean = train_features.mean(axis=0, keepdims=True)
        std = train_features.std(axis=0, keepdims=True) + 1e-5
        train_features = (train_features - mean) / std
        if fraction < 1.0:
            test_features = (test_features - mean) / std

    return train_features, test_features, train_targets, test_targets


def load_json_data(path, fraction=None, examples_per_class=None):
    """
    Loads a dataset stored as a JSON file. This will not split your dataset
    into training and testing sets, rather it returns all features and targets
    in `train_features` and `train_targets` and leaves `test_features` and
    `test_targets` as empty numpy arrays. This is done to match the API
    of the other data loaders.

    Args:
        path - (string) Path to json file containing the data
        fraction - (float) Ignored.
        examples_per_class - (int) - Ignored.

    Returns:
        train_features - (np.array) An Nxd array of features, where N is the
            number of examples and d is the number of features.
        test_features - (np.array) An empty 2D numpy array.
        train_targets - (np.array) A 1D array of targets of size N.
        test_targets - (np.array) An empty 1D array.
    """
    with open(path, 'rb') as file:
        data = json.load(file)
    features = np.array(data[0]).astype(float)
    targets = np.array(data[1]).astype(int)

    return features, np.array([[]]), targets, np.array([])


def load_mnist_data(threshold, fraction=1.0, examples_per_class=500, mnist_folder='data'):
    """
    Loads a subset of the MNIST dataset.

    Arguments:
        threshold - (int) One greater than the maximum digit in the selected
            subset. For example to get digits [0, 1, 2] this arg should be 3, or
            to get the digits [0, 1, 2, 3, 4, 5, 6] this arg should be 7.
        fraction - (float) Value between 0.0 and 1.0 representing the fraction
            of data to include in the training set. The remaining data is
            included in the test set. Unused if dataset == 'synthetic'.
        examples_per_class - (int) Number of examples to retrieve in each
            class.
        mnist_folder - (string) Path to folder contain MNIST binary files.

    Returns:
        train_features - (np.array) An Nxd array of features, where N is the
            number of examples and d is the number of features.
        test_features - (np.array) An Nxd array of features, where M is the
            number of examples and d is the number of features.
        train_targets - (np.array) A 1D array of targets of size N.
        test_targets - (np.array) A 1D array of targets of size M.
    """
    assert 0.0 <= fraction <= 1.0, 'Whoopsies! Incorrect value for fraction :P'

    train_examples = int(examples_per_class * fraction)
    if train_examples == 0:
        train_features, train_targets = np.array([[]]), np.array([])
    else:
        train_features, train_targets = _load_mnist(
            dataset='training', digits=range(threshold), path=mnist_folder)
        train_features, train_targets = stratified_subset(
            train_features, train_targets, train_examples)
        train_features = train_features.reshape((len(train_features), -1))

    test_examples = examples_per_class - train_examples
    if test_examples == 0:
        test_features, test_targets = np.array([[]]), np.array([])
    else:
        test_features, test_targets = _load_mnist(
            dataset='testing', digits=range(threshold), path=mnist_folder)
        test_features, test_targets = stratified_subset(
            test_features, test_targets, test_examples)
        test_features = test_features.reshape((len(test_features), -1))

    return train_features, test_features, train_targets, test_targets


def _load_mnist(path, dataset="training", digits=None, asbytes=False,
                selection=None, return_labels=True, return_indices=False):
    """
    Loads MNIST files into a 3D numpy array. Does not automatically download
    the dataset. You must download the dataset manually. The data can be
    downloaded from http://yann.lecun.com/exdb/mnist/.

    Examples:
        1) Assuming that you have downloaded the MNIST database in a directory
        called 'data', this will load all images and labels from the training
        set:

            images, labels = _load_mnist('training')

        2) And this will load 100 sevens from the test partition:

            sevens = _load_mnist('testing', digits=[7], selection=slice(0, 100),
                                return_labels=False)

    Arguments:
        path - (str) Path to your MNIST datafiles.
        dataset - (str) Either "training" or "testing". The data partition to
            load.
        digits - (list or None) A list of integers specifying the digits to
            load. If None, the entire database is loaded.
        asbytes - (bool) If True, returns data as ``numpy.uint8`` in [0, 255]
            as opposed to ``numpy.float64`` in [0.0, 1.0].
        selection - (slice) Using a `slice` object, specify what subset of the
            dataset to load. An example is ``slice(0, 20, 2)``, which would
            load every other digit until--but not including--the twentieth.
        return_labels - (bool) Specify whether or not labels should be
            returned. This is also a speed performance if digits are not
            specified, since then the labels file does not need to be read at
            all.
        return_indicies - (bool) Specify whether or not to return the MNIST
            indices that were fetched. This is valuable only if digits is
            specified, because in that case it can be valuable to know how far
            in the database it reached.
    Returns:
        images - (np.array) Image data of shape ``(N, rows, cols)``, where
            ``N`` is the number of images. If neither labels nor indices are
            returned, then this is returned directly, and not inside a 1-sized
            tuple.
        labels - (np.array) Array of size ``N`` describing the labels.
            Returned only if ``return_labels`` is `True`, which is default.
        indices - (np.array) The indices in the database that were returned.
    """

    # The files are assumed to have these names and should be found in 'path'
    files = {
        'training': ('train-images-idx3-ubyte', 'train-labels-idx1-ubyte'),
        'testing': ('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte'),
    }

    try:
        images_fname = os.path.join(path, files[dataset][0])
        labels_fname = os.path.join(path, files[dataset][1])
    except KeyError:
        raise ValueError("Data set must be 'testing' or 'training'")

    # We can skip the labels file only if digits aren't specified and labels
    # aren't asked for
    if return_labels or digits is not None:
        flbl = open(labels_fname, 'rb')
        magic_nr, size = struct.unpack(">II", flbl.read(8))
        labels_raw = pyarray("b", flbl.read())
        flbl.close()

    fimg = open(images_fname, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    images_raw = pyarray("B", fimg.read())
    fimg.close()

    if digits:
        indices = [k for k in range(size) if labels_raw[k] in digits]
    else:
        indices = range(size)

    if selection:
        indices = indices[selection]

    images = np.zeros((len(indices), rows, cols), dtype=np.uint8)

    if return_labels:
        labels = np.zeros((len(indices)), dtype=np.int8)
    for i in range(len(indices)):
        images[i] = np.array(images_raw[indices[i] * rows * cols:(indices[i] + 1) * rows * cols]).reshape((rows, cols))
        if return_labels:
            labels[i] = labels_raw[indices[i]]

    if not asbytes:
        images = images.astype(float)/255.0

    ret = (images,)
    if return_labels:
        ret += (labels,)
    if return_indices:
        ret += (indices,)

    if len(ret) == 1:
        return ret[0]  # Don't return a tuple of one

    return ret


def stratified_subset(features, targets, examples_per_class):
    """
    Evenly sample the dataset across unique classes. Requires each unique class
    to have at least examples_per_class examples.

    Arguments:
        features - (np.array) An Nxd array of features, where N is the
            number of examples and d is the number of features.
        targets - (np.array) A 1D array of targets of size N.
        examples_per_class - (int) The number of examples to take in each
            unique class.
    Returns:
        train_features - (np.array) An Nxd array of features, where N is the
            number of examples and d is the number of features.
        test_features - (np.array) An Nxd array of features, where M is the
            number of examples and d is the number of features.
        train_targets - (np.array) A 1D array of targets of size N.
        test_targets - (np.array) A 1D array of targets of size M.
    """
    idxs = np.array([False] * len(features))
    for target in np.unique(targets):
        idxs[np.where(targets == target)[0][:examples_per_class]] = True
    return features[idxs], targets[idxs]



class GradientDescent:
    """
    This is a linear classifier similar to the one you implemented in the
    linear regressor homework. This is the classification via regression
    case. The goal here is to learn some hyperplane, y = w^T x + b, such that
    when features, x, are processed by our model (w and b), the result is
    some value y. If y is in [0.0, +inf), the predicted classification label
    is +1 and if y is in (-inf, 0.0) the predicted classification label is
    -1.

    The catch here is that we will not be using the closed form solution,
    rather, we will be using gradient descent. In your fit function you
    will determine a loss and update your model (w and b) using gradient
    descent. More details below.

    Arguments:
        loss - (string) The loss function to use. Either 'hinge' or 'squared'.
        regularization - (string or None) The type of regularization to use.
            One of 'l1', 'l2', or None. See regularization.py for more details.
        learning_rate - (float) The size of each gradient descent update step.
        reg_param - (float) The hyperparameter that controls the amount of
            regularization to perform. Must be non-negative.
    """
    def __init__(self, loss, regularization=None,
                 learning_rate=0.01, reg_param=0.05):
        self.learning_rate = learning_rate

        # Select regularizer
        if regularization == 'l1':
            regularizer = L1Regularization(reg_param)
        elif regularization == 'l2':
            regularizer = L2Regularization(reg_param)
        elif regularization is None:
            regularizer = None
        else:
            raise ValueError(
                'Regularizer {} is not defined'.format(regularization))

        # Select loss function
        if loss == 'hinge':
            self.loss = HingeLoss(regularizer)
        elif loss == 'squared':
            self.loss = SquaredLoss(regularizer)
        else:
            raise ValueError('Loss function {} is not defined'.format(loss))

        self.model = None

    def fit(self, features, targets, batch_size=None, max_iter=1000):
        """
        Fits a gradient descent learner to the features and targets. The
        pseudocode for the fitting algorithm is as follow:
          - Initialize the model parameters to uniform random values in the
            interval [-0.1, +0.1].
          - While not converged:
            - Compute the gradient of the loss with respect to the current
              batch.
            - Update the model parameters by moving them in the direction
              opposite to the current gradient. Use the learning rate as the
              step size.
        For the convergence criteria, compute the loss over all examples. If
        this loss changes by less than 1e-4 during an update, assume that the
        model has converged. If this convergence criteria has not been met
        after max_iter iterations, also assume convergence and terminate.

        You should include a bias term by APPENDING a column of 1s to your
        feature matrix. The bias term is then the last value in self.model.

        Arguments:
            features - (np.array) An Nxd array of features, where N is the
                number of examples and d is the number of features.
            targets - (np.array) A 1D array of targets of length N.
            batch_size - (int or None) The number of examples used in each
                iteration. If None, use all of the examples in each update.
            max_iter - (int) The maximum number of updates to perform.
        Modifies:
            self.model - (np.array) A 1D array of model parameters of length
                d+1. The +1 refers to the bias term.
        """
        """
        train_features = features.copy()

        features = np.c_[features, np.ones((features.shape[0], 1))]
        
        
        w = np.random.uniform(-0.1, 0.1, features.shape[1])
        
        iterNum = []
        accur = []
        forward = []
        backward = []
        
        batch = 0
        batch_target = 0

        old = 0
        new = 0

        
                

        for e in range(1):
            if batch_size == None:
                batch = features[:]
                batch_target = targets[:]
            else:
                rand = np.random.randint(features.shape[0], size=batch_size)
                batch = features[:batch_size]
                batch_target = targets[:batch_size]
                for j in range(batch_size):
                    batch[j] = features[rand[j]]
                    batch_target[j] = targets[rand[j]]

            for i in range(max_iter):
                
                    # print("batch: ", batch.shape[0])
                old = self.loss.forward(batch, w, batch_target)

                w -= self.loss.backward(batch, w, batch_target) * self.learning_rate

                new = self.loss.forward(batch, w, batch_target)

                self.model = w
                # print(old)
                ##question1
                # predictions = self.predict(train_features)
                
                # accuracy = 0.0
                # for j in range(predictions.shape[0]):
                #     if predictions[j] == targets[j]:
                #         accuracy += 1.0
                    # accuracy += 1.0 if predictions[i] == targets[i] else 0.0
                # print("sum: ", accuracy)
                # accuracy /= predictions.shape[0]

                # iterNum.append(i)
                # accur.append(accuracy)
                # forward.append(old)
                # backward.append(new)
                # print("iteration: ", i)
                # print("accuracy: ", accuracy)

                ##question 1

                if np.abs(new - old) < 1e-4:
                    # print("iteration: ", iterNum)
                    # print("accuracy: ", accur)
                    # print("forward loss: ", forward)
                    # print("backward loss: ", backward)

                    self.model = w
                    break
            #question 1b
            predictions = self.predict(train_features)
                
            accuracy = 0.0
            for j in range(predictions.shape[0]):
                if predictions[j] == targets[j]:
                    accuracy += 1.0
            accuracy /= predictions.shape[0]
            iterNum.append(e)
            accur.append(accuracy)
            forward.append(self.loss.forward(features, w, targets))
            backward.append(new)
            
            # if np.abs(new - old) < 1e-4:
            #     print("iteration: ", iterNum)
            #     print("accuracy: ", accur)
            #     print("forward loss: ", forward)
            #     print("backward loss: ", backward)
            #question 1b


        print("iteration: ", iterNum)
        print("accuracy: ", accur)
        print("forward loss: ", forward)
        print("backward loss: ", backward)


        self.model = w
        return
        """
        """
        features = np.c_[features, np.ones((features.shape[0], 1))]
        w = np.random.uniform(-0.1, 0.1, features.shape[1])
        batch = 0
        batch_target = 0

        # print("batch_size: ", batch_size)

        if batch_size is None:
                batch = features[:]
                batch_target = targets[:]
        else:
            rand = np.random.randint(features.shape[0], size=batch_size)
            batch = features[:batch_size]
            batch_target = targets[:batch_size]
            for j in range(batch_size):
                batch[j] = features[rand[j]]
                batch_target[j] = targets[rand[j]]
        for i in range(max_iter):
            

            old = self.loss.forward(batch, w, batch_target)

            w -= self.loss.backward(batch, w, batch_target) * self.learning_rate

            new = self.loss.forward(batch, w, batch_target)

            if np.abs(new - old) < 1e-4:
                self.model = w
                return
        
        self.model = w
        return
        """
        previous_features = features.copy()
        one = np.ones((features.shape[0],1))
        features = np.c_[features,one]
        w = np.random.uniform(-0.1,0.1,features.shape[1])
        loss_history = []
        acc_history = []
        if batch_size is None:
            batch_size = features.shape[0]
        else:
            if batch_size > features.shape[0]:
                batch_size = features.shape[0]
        b = features.shape[0] // batch_size
        for i in range(max_iter):
            pre = self.loss.forward(features,w,targets)
            for j in range(b):
                index = random.sample(range(features.shape[0]),batch_size)
                temp_features = features.copy()[index]
                temp_targets = targets.copy()[index]
                w -= self.loss.backward(temp_features,w,temp_targets) * self.learning_rate
            now = self.loss.forward(features,w,targets)
            loss_history.append(now)
            self.model = w
            acc_history.append(np.mean(self.predict(previous_features) == targets))
            if np.abs(now - pre) < 1e-4:
                break
        return loss_history,acc_history


    def predict(self, features):
        """
        Predicts the class labels of each example in features. Model output
        values at and above 0 are predicted to have label +1. Non-positive
        output values are predicted to have label -1.

        NOTE: your predict function should make use of your confidence
        function (see below).

        Arguments:
            features - (np.array) A Nxd array of features, where N is the
                number of examples and d is the number of features.
        Returns:
            predictions - (np.array) A 1D array of predictions of length N,
                where index d corresponds to the prediction of row N of
                features.
        """
        """
        features = np.c_[features, np.ones((features.shape[0], 1))]

        pred = np.dot(features, self.model)

        for i in range(features.shape[0]):
            pred[i] = 1 if pred[i] >= 0 else -1

        return pred
        """
        one = np.ones((features.shape[0],1))
        features = np.c_[features,one]
        pre = features.dot(self.model)
        for i in range(features.shape[0]):
            if pre[i] >= 0:
                pre[i] = 1
            else:
                pre[i] = -1
        return pre

    def confidence(self, features):
        """
        Returns the raw model output of the prediction. In other words, rather
        than predicting +1 for values above 0 and -1 for other values, this
        function returns the original, unquantized value.

        Arguments:
            features - (np.array) A Nxd array of features, where N is the
                number of examples and d is the number of features.
        Returns:
            confidence - (np.array) A 1D array of confidence values of length
                N, where index d corresponds to the confidence of row N of
                features.
        """
        
        features = np.c_[features, np.ones((features.shape[0], 1))]

        return np.dot(features, self.model)


def question1a():
    train_features, test_features, train_targets, test_targets = load_data('mnist-binary', fraction=1, base_folder='.')
    learner = GradientDescent('hinge', learning_rate=1e-4)

    learner.fit(train_features, train_targets)

def visual1a():

    

    iterNum = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 857, 858, 859, 860, 861, 862, 863, 864, 865, 866, 867, 868, 869, 870, 871, 872]
    accuracy =  [0.816, 0.816, 0.816, 0.816, 0.817, 0.818, 0.818, 0.82, 0.82, 0.82, 0.823, 0.823, 0.824, 0.825, 0.825, 0.828, 0.829, 0.83, 0.83, 0.83, 0.83, 0.831, 0.831, 0.831, 0.832, 0.832, 0.833, 0.834, 0.834, 0.836, 0.839, 0.841, 0.843, 0.844, 0.845, 0.845, 0.846, 0.847, 0.847, 0.848, 0.848, 0.848, 0.848, 0.849, 0.85, 0.85, 0.852, 0.852, 0.853, 0.853, 0.854, 0.856, 0.857, 0.857, 0.857, 0.858, 0.858, 0.858, 0.858, 0.861, 0.861, 0.861, 0.861, 0.861, 0.862, 0.862, 0.863, 0.863, 0.863, 0.864, 0.864, 0.864, 0.864, 0.865, 0.867, 0.87, 0.871, 0.871, 0.871, 0.871, 0.871, 0.871, 0.872, 0.872, 0.872, 0.872, 0.873, 0.873, 0.873, 0.873, 0.873, 0.873, 0.873, 0.874, 0.874, 0.875, 0.875, 0.876, 0.876, 0.876, 0.877, 0.877, 0.877, 0.877, 0.877, 0.877, 0.877, 0.877, 0.877, 0.877, 0.878, 0.878, 0.878, 0.878, 0.878, 0.878, 0.879, 0.879, 0.88, 0.881, 0.881, 0.881, 0.881, 0.881, 0.882, 0.882, 0.883, 0.883, 0.884, 0.884, 0.884, 0.884, 0.884, 0.884, 0.884, 0.884, 0.884, 0.885, 0.886, 0.886, 0.886, 0.886, 0.886, 0.886, 0.887, 0.887, 0.887, 0.887, 0.887, 0.887, 0.887, 0.887, 0.888, 0.888, 0.888, 0.889, 0.889, 0.889, 0.889, 0.889, 0.889, 0.89, 0.89, 0.89, 0.891, 0.891, 0.891, 0.891, 0.891, 0.891, 0.891, 0.891, 0.891, 0.891, 0.891, 0.891, 0.891, 0.891, 0.893, 0.893, 0.893, 0.893, 0.893, 0.893, 0.893, 0.894, 0.894, 0.894, 0.894, 0.894, 0.894, 0.895, 0.895, 0.895, 0.895, 0.895, 0.895, 0.895, 0.895, 0.895, 0.895, 0.896, 0.896, 0.896, 0.896, 0.896, 0.896, 0.896, 0.896, 0.896, 0.896, 0.897, 0.898, 0.898, 0.898, 0.898, 0.898, 0.898, 0.898, 0.898, 0.899, 0.899, 0.899, 0.899, 0.899, 0.899, 0.899, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.901, 0.901, 0.901, 0.901, 0.901, 0.901, 0.901, 0.901, 0.901, 0.901, 0.901, 0.901, 0.901, 0.901, 0.901, 0.901, 0.901, 0.901, 0.901, 0.901, 0.901, 0.901, 0.901, 0.901, 0.901, 0.902, 0.902, 0.902, 0.903, 0.903, 0.903, 0.903, 0.903, 0.903, 0.903, 0.904, 0.904, 0.904, 0.904, 0.904, 0.904, 0.904, 0.905, 0.905, 0.905, 0.905, 0.905, 0.905, 0.907, 0.907, 0.907, 0.907, 0.907, 0.907, 0.907, 0.907, 0.907, 0.907, 0.907, 0.907, 0.907, 0.907, 0.907, 0.908, 0.91, 0.91, 0.91, 0.91, 0.91, 0.91, 0.91, 0.91, 0.91, 0.91, 0.91, 0.91, 0.91, 0.91, 0.91, 0.91, 0.91, 0.91, 0.91, 0.91, 0.91, 0.91, 0.91, 0.91, 0.91, 0.91, 0.91, 0.91, 0.91, 0.91, 0.91, 0.91, 0.911, 0.911, 0.911, 0.911, 0.911, 0.911, 0.912, 0.912, 0.912, 0.912, 0.912, 0.912, 0.913, 0.913, 0.913, 0.913, 0.914, 0.914, 0.914, 0.914, 0.914, 0.914, 0.914, 0.914, 0.914, 0.915, 0.915, 0.915, 0.915, 0.915, 0.915, 0.915, 0.915, 0.915, 0.915, 0.915, 0.915, 0.916, 0.916, 0.916, 0.916, 0.917, 0.917, 0.917, 0.917, 0.917, 0.917, 0.917, 0.917, 0.919, 0.919, 0.919, 0.919, 0.919, 0.919, 0.919, 0.919, 0.919, 0.919, 0.919, 0.919, 0.919, 0.919, 0.919, 0.92, 0.92, 0.92, 0.92, 0.92, 0.92, 0.92, 0.92, 0.92, 0.921, 0.921, 0.921, 0.921, 0.921, 0.921, 0.921, 0.921, 0.921, 0.921, 0.921, 0.921, 0.921, 0.921, 0.921, 0.921, 0.921, 0.921, 0.921, 0.921, 0.922, 0.922, 0.922, 0.922, 0.922, 0.922, 0.922, 0.922, 0.922, 0.922, 0.922, 0.922, 0.922, 0.922, 0.922, 0.922, 0.923, 0.923, 0.923, 0.923, 0.923, 0.923, 0.923, 0.923, 0.923, 0.923, 0.923, 0.923, 0.923, 0.923, 0.924, 0.924, 0.924, 0.924, 0.924, 0.924, 0.925, 0.925, 0.925, 0.925, 0.925, 0.925, 0.925, 0.925, 0.926, 0.927, 0.927, 0.927, 0.927, 0.927, 0.927, 0.927, 0.927, 0.927, 0.928, 0.928, 0.928, 0.928, 0.928, 0.928, 0.928, 0.928, 0.928, 0.928, 0.928, 0.928, 0.928, 0.928, 0.928, 0.928, 0.928, 0.928, 0.928, 0.928, 0.928, 0.928, 0.928, 0.929, 0.929, 0.929, 0.929, 0.929, 0.929, 0.929, 0.929, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.931, 0.931, 0.931, 0.931, 0.931, 0.931, 0.931, 0.932, 0.932, 0.932, 0.932, 0.932, 0.933, 0.933, 0.933, 0.933, 0.933, 0.933, 0.933, 0.933, 0.933, 0.933, 0.933, 0.933, 0.933, 0.933, 0.934, 0.934, 0.934, 0.934, 0.934, 0.934, 0.935, 0.935, 0.935, 0.935, 0.935, 0.935, 0.935, 0.935, 0.935, 0.935, 0.935, 0.935, 0.935, 0.935, 0.935, 0.935, 0.935, 0.935, 0.935, 0.935, 0.935, 0.935, 0.935, 0.936, 0.936, 0.936, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.937, 0.938, 0.938, 0.938, 0.938, 0.938, 0.938, 0.938, 0.938, 0.938, 0.939, 0.939, 0.939, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.941, 0.941, 0.941, 0.941, 0.941, 0.941, 0.941, 0.941, 0.941, 0.941, 0.941, 0.941, 0.941, 0.941, 0.941, 0.941, 0.942, 0.942, 0.943, 0.943, 0.944, 0.944, 0.944, 0.944, 0.944, 0.944, 0.945, 0.945, 0.945, 0.945, 0.945, 0.945, 0.945, 0.945, 0.945, 0.945, 0.946, 0.946, 0.946, 0.946, 0.946, 0.947, 0.947, 0.947, 0.947, 0.947, 0.947, 0.947, 0.947, 0.947, 0.947, 0.947, 0.947, 0.947, 0.948, 0.949, 0.949, 0.949, 0.949, 0.949, 0.949, 0.949, 0.949, 0.949, 0.949, 0.949, 0.949, 0.949, 0.949, 0.949, 0.949, 0.949, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95]
    forward = [0.530577661440502, 0.5279503116078128, 0.5253341362355305, 0.5227189979685496, 0.5201168117522096, 0.517530595597534, 0.5149708316176308, 0.5124429525383776, 0.5099220160593508, 0.5074059183280523, 0.5049049447475045, 0.5024150418683783, 0.49994208411837293, 0.4974856150247064, 0.4950514961244811, 0.4926484771807653, 0.4902625193078092, 0.48790253125654304, 0.48554458744925255, 0.4832033369800912, 0.48088796698253317, 0.4785988125472395, 0.47635161669852316, 0.47412053165408485, 0.4719067570157801, 0.4697187673067844, 0.4675450085486, 0.4653748593894711, 0.46320774176305496, 0.46105161903156033, 0.45891252906783137, 0.4568051568789358, 0.4547170122992071, 0.45265823193406823, 0.4506322946138546, 0.4486431313160963, 0.4466829516502423, 0.44474116091943694, 0.44281828066444057, 0.44091641432873985, 0.43902331795097327, 0.4371393184656894, 0.4352614357721478, 0.4334006622740526, 0.43154188036462954, 0.4297026075208031, 0.427877693120041, 0.4260869893692759, 0.42431837386692195, 0.42257902464388436, 0.42087358984321627, 0.4191892129567013, 0.41752744423559285, 0.41586567551448456, 0.41421352123229055, 0.4125764730818097, 0.4109449544972565, 0.40931824055314686, 0.407691526609037, 0.40607808283610175, 0.40449319841980336, 0.402914909105807, 0.40133835161264536, 0.3997678067991393, 0.3981983127222346, 0.39664599640996134, 0.39509942804277054, 0.3935541028343865, 0.39201840636162966, 0.39050152471743277, 0.38900421399752977, 0.3875315586242346, 0.38606586116294045, 0.3846068265430399, 0.3831580294188625, 0.38171745638074384, 0.3802939132655314, 0.37888265689543943, 0.37748656877450737, 0.3760947602880011, 0.3747029518014951, 0.37331867442223854, 0.37194526410030543, 0.3705844973438213, 0.36924610193449436, 0.3679206391700317, 0.3666259609873265, 0.36534729111192005, 0.3640745827038597, 0.3628116283670289, 0.3615655703655846, 0.3603222361833183, 0.35909417900070634, 0.3578790818779644, 0.35668008514436034, 0.3554946724124259, 0.3543249358969668, 0.3531574500882079, 0.3520028196261622, 0.3508624856192853, 0.3497361846938716, 0.34861724896597807, 0.34751009299715235, 0.3464085312679891, 0.34531354321516233, 0.3442253560720423, 0.3431424824654166, 0.34206622793935626, 0.3410124276294852, 0.3399704187302868, 0.33892840983108796, 0.3378887828877452, 0.336858138503111, 0.33583109669635636, 0.33481201488693363, 0.3338050908923697, 0.3328103860395237, 0.3318204415864648, 0.3308392745966906, 0.32986647282507797, 0.3289063568546877, 0.32794805054618126, 0.327002348902306, 0.3260566472584311, 0.32511303596005386, 0.324186124426291, 0.32326773297923717, 0.3223583841048924, 0.32145475294600284, 0.32056527906198756, 0.3196858476243988, 0.3188173797038282, 0.31796073734223584, 0.3171057314462725, 0.3162576932197143, 0.31541574775007, 0.31458338688013027, 0.31376280298996, 0.3129572146286041, 0.31215481671326034, 0.31135241879791675, 0.3105500208825729, 0.3097481357033351, 0.3089525969589188, 0.3081597512883765, 0.30739140499645723, 0.30663524692768745, 0.3058790888589183, 0.30512426202723814, 0.30437561147963627, 0.30362908117584586, 0.3028848802239396, 0.3021467596471394, 0.30141573912352787, 0.30068471859991586, 0.2999536980763039, 0.29922267755269233, 0.2984931921850575, 0.29777378185978925, 0.2970586427336186, 0.2963494067957048, 0.2956401708577909, 0.2949322888659672, 0.29422827517158967, 0.29353054034205295, 0.2928386013061005, 0.2921508160029418, 0.29146317282736856, 0.29078026093636034, 0.2900974591425449, 0.28942086436681747, 0.28875327435212605, 0.2880917051620795, 0.2874317503479956, 0.286781375982879, 0.2861426043498277, 0.2855038327167764, 0.284865061083725, 0.28423040244886044, 0.2836006493911474, 0.28297285012803286, 0.2823519671613483, 0.2817311751714618, 0.28111994917498495, 0.2805137878771959, 0.27990762657940677, 0.2793022759734545, 0.2787096679505027, 0.27812676452636204, 0.2775569355934899, 0.2769884601309855, 0.27642576508188493, 0.27587099639098833, 0.27532280077352755, 0.2747778660099616, 0.2742453181436902, 0.2737127702774184, 0.27318022241114726, 0.2726495927115092, 0.27212121579939647, 0.2715928388872836, 0.2710644619751705, 0.27053608506305754, 0.2700090429930433, 0.26948879407334647, 0.2689719851644037, 0.2684551762554608, 0.2679393001809614, 0.26743052018948754, 0.2669217401980134, 0.26641296020653926, 0.26590615391147754, 0.26540138415806447, 0.2649002744052298, 0.2644037087638612, 0.2639094114583576, 0.2634183990712352, 0.26292738668411253, 0.26243768235530157, 0.2619551231811531, 0.2614746146918097, 0.26099682650616823, 0.2605240221155574, 0.26005712272186293, 0.25959231806342775, 0.2591360878858898, 0.25868143134023136, 0.2582270942935258, 0.25777824622180645, 0.2573307761678776, 0.25688772446888214, 0.2564502274700065, 0.25601408310481455, 0.25558094671987885, 0.2551508390140313, 0.2547210563569271, 0.2542962630353675, 0.2538734801951422, 0.253453525660757, 0.2530380046699308, 0.25262425009127243, 0.25221049551261376, 0.25179852886348497, 0.25139188246391064, 0.25098753245466, 0.25058318244540967, 0.25017883243615885, 0.2497744824269081, 0.2493713299820246, 0.24897053633886476, 0.24857246224525892, 0.24817438815165344, 0.24777885485276627, 0.24738775936440635, 0.24699941481128498, 0.2466155743707137, 0.24623374780686322, 0.24585346009406614, 0.2454751733189997, 0.2450983335276911, 0.2447232105741474, 0.2443480876206034, 0.2439747755774111, 0.24360341772773944, 0.24323205987806804, 0.24286070202839602, 0.24248946918499242, 0.24212047079046387, 0.24175147239593534, 0.24138247400140705, 0.24101347560687836, 0.24064456714867677, 0.24027857313717574, 0.23991424105091388, 0.23955611088005233, 0.23920577517918523, 0.23886284761676543, 0.23851992005434577, 0.23817699249192587, 0.23783406492950612, 0.2374911373670865, 0.2371482098046667, 0.23680528224224698, 0.23646339843865857, 0.2361260861137536, 0.23579136106365056, 0.235456636013548, 0.2351233526854203, 0.23479164369013647, 0.2344621135781241, 0.23413588788670447, 0.23381222084932116, 0.2334885538119381, 0.23316488677455494, 0.23284238555305625, 0.23252140352453946, 0.2322004214960229, 0.23187943946750622, 0.23155845743898973, 0.23123808294050532, 0.23092008936973618, 0.23060209579896668, 0.23028410222819748, 0.229967518537851, 0.22965246997974065, 0.2293378113757845, 0.22902615407244464, 0.22871929957776063, 0.22841244508307654, 0.22810559058839225, 0.2277996543194802, 0.22749558060216024, 0.22719303632341337, 0.2268932635372859, 0.22659349075115823, 0.22629371796503064, 0.2259939451789029, 0.22569417239277528, 0.22539439960664778, 0.22509462682052023, 0.22479485403439253, 0.22449508124826484, 0.22419530846213745, 0.22389553567600992, 0.22359576288988237, 0.22329599010375467, 0.22299621731762703, 0.22269644453149945, 0.2223966717453718, 0.2220972457084607, 0.2218004351792078, 0.22150362464995488, 0.22120681412070187, 0.2209100035914488, 0.2206131930621959, 0.22031638253294303, 0.22002041873403744, 0.21972831613942503, 0.21943621354481282, 0.21914546828375595, 0.21885702283750155, 0.21857034613928508, 0.21828366944106883, 0.2179979516850024, 0.21771405229939622, 0.2174301529137899, 0.2171472202995229, 0.2168673291223431, 0.21658894814895158, 0.21631056717556002, 0.21603218620216846, 0.21575447727034314, 0.21547774468752212, 0.21520276727910218, 0.2149320202458516, 0.21466298473072415, 0.21439512249093615, 0.21413235552738458, 0.2138715391491546, 0.2136107227709251, 0.2133499063926954, 0.21309075021461915, 0.21283528827601994, 0.2125838117720995, 0.21233688229948025, 0.21209084619196636, 0.21184834184912807, 0.21160583750628992, 0.2113633331634517, 0.21112082882061337, 0.21087844210904702, 0.21063845429777467, 0.21039846648650234, 0.21015847867522977, 0.2099184908639573, 0.20967850305268504, 0.2094385152414124, 0.20919852743013997, 0.2089585396188676, 0.20871855180759535, 0.20847856399632278, 0.2082385761850504, 0.20799858837377802, 0.2077586005625056, 0.20751861275123312, 0.20727862493996072, 0.20703863712868809, 0.20680193409672895, 0.20657000242909282, 0.20633807076145674, 0.20610613909382056, 0.2058742074261846, 0.2056422757585484, 0.20541034409091227, 0.20517968578379267, 0.20495202304966, 0.20472436031552738, 0.20449669758139466, 0.20426903484726197, 0.2040413721131293, 0.20381370937899673, 0.20358619050986204, 0.20336175064769146, 0.20313950209337403, 0.20291748232183476, 0.20270106550365935, 0.20248694997188857, 0.20227283444011762, 0.20205871890834687, 0.201844603376576, 0.20163048784480517, 0.20141746539192767, 0.20120563974957692, 0.2009938141072262, 0.20078198846487558, 0.20057016282252513, 0.2003583371801745, 0.20014651153782384, 0.19993468589547322, 0.19972286025312255, 0.1995125162034825, 0.19930420518059278, 0.19909589415770282, 0.19888758313481314, 0.19867927211192316, 0.19847096108903337, 0.19826376304220977, 0.19805816325815953, 0.1978525634741096, 0.1976479028469825, 0.19744453105516457, 0.19724115926334673, 0.1970377874715286, 0.1968344156797107, 0.1966310438878929, 0.19642767209607487, 0.19622430030425694, 0.19602092851243905, 0.19581755672062112, 0.19561418492880323, 0.1954108131369854, 0.19520744134516738, 0.19500406955334948, 0.19480069776153153, 0.19459732596971352, 0.19439395417789554, 0.19419058238607775, 0.1939872252962926, 0.19378394834291665, 0.19358067138954083, 0.19337739443616495, 0.19317411748278907, 0.1929708405294132, 0.19276756357603728, 0.1925642866226616, 0.19236100966928557, 0.19215773271590983, 0.19195445576253392, 0.19175194854016422, 0.19155126284219273, 0.19135057714422118, 0.19114989144624953, 0.1909492057482779, 0.1907485200503062, 0.19054783435233455, 0.19034714865436317, 0.19014646295639143, 0.18994614579900895, 0.18974776860292814, 0.18954939140684715, 0.18935101421076636, 0.18915263701468552, 0.1889542598186048, 0.18875588262252402, 0.18855757626631134, 0.18836095397815983, 0.18816433169000815, 0.18796770940185664, 0.18777164610571861, 0.18757678300246158, 0.1873819198992049, 0.18718705679594796, 0.18699219369269113, 0.18679733058943418, 0.1866024674861774, 0.1864076043829205, 0.18621274127966367, 0.18601899998876106, 0.18582693451258153, 0.18563486903640233, 0.18544280356022316, 0.18525073808404388, 0.18505867260786463, 0.1848686604112746, 0.18468254297454265, 0.1844983258495525, 0.18431479206661697, 0.1841326852035097, 0.18395261916863442, 0.18377255313375898, 0.18359248709888365, 0.18341242106400837, 0.18323235502913288, 0.18305278133830996, 0.18287575618563087, 0.1827015622500054, 0.18252736831438, 0.18235317437875434, 0.18217898044312905, 0.1820047865075036, 0.18183059257187822, 0.1816576854683348, 0.18148601728309724, 0.1813158278119727, 0.1811477814506901, 0.18097973508940748, 0.18081209105376383, 0.18064639019180806, 0.1804818850779467, 0.18031737996408548, 0.1801531290664119, 0.17999027072623966, 0.1798274123860674, 0.17966455404589518, 0.17950169570572297, 0.17933934469623866, 0.17917747067066084, 0.17901559664508293, 0.17885372261950488, 0.17869184859392687, 0.17852997456834885, 0.178368100542771, 0.17820622651719312, 0.17804435249161507, 0.1778824784660372, 0.17772060444045923, 0.17755873041488127, 0.17739685638930322, 0.17723498236372542, 0.17707310833814757, 0.1769112343125694, 0.17674936028699145, 0.17658748626141355, 0.1764256122358355, 0.17626373821025756, 0.17610186418467985, 0.1759399901591017, 0.17577811613352376, 0.17561624210794588, 0.17545436808236783, 0.1752924940567899, 0.175130620031212, 0.1749687460056339, 0.17480687198005604, 0.17464499795447808, 0.17448312392890022, 0.17432124990332232, 0.1741593758777442, 0.17399750185216634, 0.1738356278265883, 0.1736737538010103, 0.17351305697461955, 0.17335416037513487, 0.17319526377565034, 0.17303636717616558, 0.172877470576681, 0.17271857397719628, 0.17255967737771175, 0.17240078077822701, 0.1722418841787423, 0.17208475364583017, 0.17193083614760502, 0.17177691864938008, 0.171623001151155, 0.17146908365292995, 0.1713151661547049, 0.17116124865647986, 0.1710079749668606, 0.1708555179886823, 0.1707038010029946, 0.17055462938919427, 0.1704063982449873, 0.1702605546496202, 0.1701154713221819, 0.16997157269487584, 0.16982767406756977, 0.16968377544026378, 0.16953987681295765, 0.16939597818565158, 0.16925207955834556, 0.16910818093103944, 0.1689642823037334, 0.1688203836764274, 0.16867658509812314, 0.16853489268588692, 0.1683932002736505, 0.16825150786141416, 0.16810981544917789, 0.16796812303694145, 0.1678264306247051, 0.16768473821246868, 0.1675430458002323, 0.16740135338799603, 0.16725966097575962, 0.16711796856352323, 0.16697627615128693, 0.16683458373905055, 0.16669289132681425, 0.1665511989145779, 0.16640950650234163, 0.16626781409010508, 0.16612673231059039, 0.1659863228858182, 0.16584591346104588, 0.16570550403627382, 0.16556509461150148, 0.1654246851867294, 0.16528505696372095, 0.16514811451350644, 0.1650111720632922, 0.16487422961307777, 0.16473728716286345, 0.1646003447126491, 0.16446340226243453, 0.16432658756650934, 0.16419157907989654, 0.164056570593284, 0.16392156210667122, 0.16378655362005853, 0.16365154513344582, 0.1635167866268276, 0.163383595760685, 0.16325119057949697, 0.16311881546005388, 0.16298715067160569, 0.16285660337843647, 0.16272605608526713, 0.16259550879209803, 0.16246496149892864, 0.16233441420575925, 0.16220386691259, 0.16207331961942073, 0.16194277232625148, 0.1618122250330821, 0.16168167773991274, 0.1615511304467436, 0.16142058315357427, 0.161290035860405, 0.1611594885672358, 0.16102894127406642, 0.16089839398089717, 0.16076784668772784, 0.16063860650837838, 0.16051114678008083, 0.1603836870517832, 0.16025622732348568, 0.1601287675951881, 0.1600013078668906, 0.15987384813859307, 0.1597463884102954, 0.15961979631276946, 0.15949546487691535, 0.15937113344106113, 0.1592473856857266, 0.15912503431377462, 0.15900268294182265, 0.15888033156987064, 0.15875798019791865, 0.15863562882596652, 0.15851327745401436, 0.15839117010497059, 0.15827005941169753, 0.15814894871842464, 0.15802783802515166, 0.15790672733187863, 0.1577856166386056, 0.1576645059453326, 0.15754339525205968, 0.15742228455878682, 0.15730117386551373, 0.15718006317224076, 0.1570589524789679, 0.15693784178569484, 0.1568167310924218, 0.15669562039914894, 0.15657450970587583, 0.156453399012603, 0.1563322883193299, 0.156211177626057, 0.15609006693278385, 0.15596895623951093, 0.15584784554623796, 0.155726734852965, 0.15560562415969206, 0.15548451346641917, 0.1553635586479447, 0.15524362100505587, 0.1551236833621672, 0.15500374571927825, 0.15488380807638955, 0.15476387043350068, 0.1546439327906119, 0.15452399514772314, 0.1544040575048344, 0.15428411986194565, 0.15416419206370616, 0.15404538785445873, 0.15392658364521145, 0.15380777943596405, 0.1536889752267167, 0.15357017101746942, 0.1534513668082221, 0.15333256259897474, 0.1532137583897274, 0.15309495418048003, 0.15297614997123282, 0.15285845040032361, 0.1527425716444929, 0.15262669288866196, 0.15251081413283119, 0.1523949353770004, 0.1522790566211695, 0.15216317786533878, 0.15204729910950798, 0.15193142035367704, 0.15181554159784622, 0.15169966284201547, 0.15158378408618453, 0.15146790533035379, 0.151352026574523, 0.15123614781869207, 0.15112026906286133, 0.1510043903070306, 0.1508887723627998, 0.15077449544067636, 0.15066021851855302, 0.1505459415964297, 0.1504316646743064, 0.150317387752183, 0.15020311083005972, 0.15008883390793637, 0.14997455698581297, 0.14986028006368957, 0.14974600314156636, 0.1496317262194429, 0.14951744929731967, 0.14940317237519624, 0.14928889545307283, 0.14917461853094968, 0.14906034160882628, 0.14894606468670293, 0.14883178776457962, 0.14871810401137708, 0.14860547546078984, 0.1484928469102026, 0.14838021835961535, 0.1482675898090282, 0.14815496125844088, 0.14804233270785364, 0.1479297041572665, 0.14781707560667934, 0.14770444705609198, 0.14759181850550487, 0.1474797633012156, 0.14736900987315113, 0.1472582564450867, 0.1471475030170222, 0.14703674958895777, 0.14692599616089325, 0.1468152427328288, 0.1467044893047645, 0.14659373587670005, 0.1464835476276687, 0.14637473987502284, 0.14626593212237693, 0.1461571243697312, 0.1460483166170852, 0.14593950886443932, 0.1458307011117934, 0.14572189335914754, 0.1456130856065017, 0.1455042778538559, 0.1453954701012099, 0.14528666234856405, 0.14517785459591823, 0.14506904684327235, 0.14496023909062639, 0.14485143133798062, 0.14474262358533474, 0.14463505582815409, 0.1445291257656036, 0.14442319570305315, 0.14431726564050276, 0.1442113355779523, 0.1441054055154018, 0.1439994754528513, 0.14389354539030091, 0.14378761532775045, 0.14368168526519998, 0.14357575520264965, 0.14346982514009912, 0.14336389507754865, 0.14325796501499818, 0.14315203495244772, 0.14304610488989727, 0.1429401748273468, 0.14283424476479634, 0.14272831470224595, 0.14262238463969548, 0.14251645457714499, 0.14241052451459454, 0.14230459445204413, 0.1421987684283742, 0.14209398561976, 0.14198966883941777, 0.14188696271381968, 0.14178425658822147, 0.1416816822059964, 0.141580825166089, 0.14147996812618163, 0.14137911108627416, 0.14127825404636662, 0.14117739700645923, 0.14107653996655184, 0.1409756829266445, 0.14087482588673697, 0.14077396884682955, 0.14067311180692216, 0.14057225476701465, 0.14047139772710732, 0.14037054068719984, 0.14026968364729245, 0.14016882660738508, 0.14006796956747758, 0.13996711252757021, 0.13986625548766277, 0.13976539844775532, 0.13966454140784784, 0.13956368436794048, 0.13946282732803308, 0.13936197028812566, 0.1392611132482182]
    backward = [0.5279503116078128, 0.5253341362355305, 0.5227189979685496, 0.5201168117522096, 0.517530595597534, 0.5149708316176308, 0.5124429525383776, 0.5099220160593508, 0.5074059183280523, 0.5049049447475045, 0.5024150418683783, 0.49994208411837293, 0.4974856150247064, 0.4950514961244811, 0.4926484771807653, 0.4902625193078092, 0.48790253125654304, 0.48554458744925255, 0.4832033369800912, 0.48088796698253317, 0.4785988125472395, 0.47635161669852316, 0.47412053165408485, 0.4719067570157801, 0.4697187673067844, 0.4675450085486, 0.4653748593894711, 0.46320774176305496, 0.46105161903156033, 0.45891252906783137, 0.4568051568789358, 0.4547170122992071, 0.45265823193406823, 0.4506322946138546, 0.4486431313160963, 0.4466829516502423, 0.44474116091943694, 0.44281828066444057, 0.44091641432873985, 0.43902331795097327, 0.4371393184656894, 0.4352614357721478, 0.4334006622740526, 0.43154188036462954, 0.4297026075208031, 0.427877693120041, 0.4260869893692759, 0.42431837386692195, 0.42257902464388436, 0.42087358984321627, 0.4191892129567013, 0.41752744423559285, 0.41586567551448456, 0.41421352123229055, 0.4125764730818097, 0.4109449544972565, 0.40931824055314686, 0.407691526609037, 0.40607808283610175, 0.40449319841980336, 0.402914909105807, 0.40133835161264536, 0.3997678067991393, 0.3981983127222346, 0.39664599640996134, 0.39509942804277054, 0.3935541028343865, 0.39201840636162966, 0.39050152471743277, 0.38900421399752977, 0.3875315586242346, 0.38606586116294045, 0.3846068265430399, 0.3831580294188625, 0.38171745638074384, 0.3802939132655314, 0.37888265689543943, 0.37748656877450737, 0.3760947602880011, 0.3747029518014951, 0.37331867442223854, 0.37194526410030543, 0.3705844973438213, 0.36924610193449436, 0.3679206391700317, 0.3666259609873265, 0.36534729111192005, 0.3640745827038597, 0.3628116283670289, 0.3615655703655846, 0.3603222361833183, 0.35909417900070634, 0.3578790818779644, 0.35668008514436034, 0.3554946724124259, 0.3543249358969668, 0.3531574500882079, 0.3520028196261622, 0.3508624856192853, 0.3497361846938716, 0.34861724896597807, 0.34751009299715235, 0.3464085312679891, 0.34531354321516233, 0.3442253560720423, 0.3431424824654166, 0.34206622793935626, 0.3410124276294852, 0.3399704187302868, 0.33892840983108796, 0.3378887828877452, 0.336858138503111, 0.33583109669635636, 0.33481201488693363, 0.3338050908923697, 0.3328103860395237, 0.3318204415864648, 0.3308392745966906, 0.32986647282507797, 0.3289063568546877, 0.32794805054618126, 0.327002348902306, 0.3260566472584311, 0.32511303596005386, 0.324186124426291, 0.32326773297923717, 0.3223583841048924, 0.32145475294600284, 0.32056527906198756, 0.3196858476243988, 0.3188173797038282, 0.31796073734223584, 0.3171057314462725, 0.3162576932197143, 0.31541574775007, 0.31458338688013027, 0.31376280298996, 0.3129572146286041, 0.31215481671326034, 0.31135241879791675, 0.3105500208825729, 0.3097481357033351, 0.3089525969589188, 0.3081597512883765, 0.30739140499645723, 0.30663524692768745, 0.3058790888589183, 0.30512426202723814, 0.30437561147963627, 0.30362908117584586, 0.3028848802239396, 0.3021467596471394, 0.30141573912352787, 0.30068471859991586, 0.2999536980763039, 0.29922267755269233, 0.2984931921850575, 0.29777378185978925, 0.2970586427336186, 0.2963494067957048, 0.2956401708577909, 0.2949322888659672, 0.29422827517158967, 0.29353054034205295, 0.2928386013061005, 0.2921508160029418, 0.29146317282736856, 0.29078026093636034, 0.2900974591425449, 0.28942086436681747, 0.28875327435212605, 0.2880917051620795, 0.2874317503479956, 0.286781375982879, 0.2861426043498277, 0.2855038327167764, 0.284865061083725, 0.28423040244886044, 0.2836006493911474, 0.28297285012803286, 0.2823519671613483, 0.2817311751714618, 0.28111994917498495, 0.2805137878771959, 0.27990762657940677, 0.2793022759734545, 0.2787096679505027, 0.27812676452636204, 0.2775569355934899, 0.2769884601309855, 0.27642576508188493, 0.27587099639098833, 0.27532280077352755, 0.2747778660099616, 0.2742453181436902, 0.2737127702774184, 0.27318022241114726, 0.2726495927115092, 0.27212121579939647, 0.2715928388872836, 0.2710644619751705, 0.27053608506305754, 0.2700090429930433, 0.26948879407334647, 0.2689719851644037, 0.2684551762554608, 0.2679393001809614, 0.26743052018948754, 0.2669217401980134, 0.26641296020653926, 0.26590615391147754, 0.26540138415806447, 0.2649002744052298, 0.2644037087638612, 0.2639094114583576, 0.2634183990712352, 0.26292738668411253, 0.26243768235530157, 0.2619551231811531, 0.2614746146918097, 0.26099682650616823, 0.2605240221155574, 0.26005712272186293, 0.25959231806342775, 0.2591360878858898, 0.25868143134023136, 0.2582270942935258, 0.25777824622180645, 0.2573307761678776, 0.25688772446888214, 0.2564502274700065, 0.25601408310481455, 0.25558094671987885, 0.2551508390140313, 0.2547210563569271, 0.2542962630353675, 0.2538734801951422, 0.253453525660757, 0.2530380046699308, 0.25262425009127243, 0.25221049551261376, 0.25179852886348497, 0.25139188246391064, 0.25098753245466, 0.25058318244540967, 0.25017883243615885, 0.2497744824269081, 0.2493713299820246, 0.24897053633886476, 0.24857246224525892, 0.24817438815165344, 0.24777885485276627, 0.24738775936440635, 0.24699941481128498, 0.2466155743707137, 0.24623374780686322, 0.24585346009406614, 0.2454751733189997, 0.2450983335276911, 0.2447232105741474, 0.2443480876206034, 0.2439747755774111, 0.24360341772773944, 0.24323205987806804, 0.24286070202839602, 0.24248946918499242, 0.24212047079046387, 0.24175147239593534, 0.24138247400140705, 0.24101347560687836, 0.24064456714867677, 0.24027857313717574, 0.23991424105091388, 0.23955611088005233, 0.23920577517918523, 0.23886284761676543, 0.23851992005434577, 0.23817699249192587, 0.23783406492950612, 0.2374911373670865, 0.2371482098046667, 0.23680528224224698, 0.23646339843865857, 0.2361260861137536, 0.23579136106365056, 0.235456636013548, 0.2351233526854203, 0.23479164369013647, 0.2344621135781241, 0.23413588788670447, 0.23381222084932116, 0.2334885538119381, 0.23316488677455494, 0.23284238555305625, 0.23252140352453946, 0.2322004214960229, 0.23187943946750622, 0.23155845743898973, 0.23123808294050532, 0.23092008936973618, 0.23060209579896668, 0.23028410222819748, 0.229967518537851, 0.22965246997974065, 0.2293378113757845, 0.22902615407244464, 0.22871929957776063, 0.22841244508307654, 0.22810559058839225, 0.2277996543194802, 0.22749558060216024, 0.22719303632341337, 0.2268932635372859, 0.22659349075115823, 0.22629371796503064, 0.2259939451789029, 0.22569417239277528, 0.22539439960664778, 0.22509462682052023, 0.22479485403439253, 0.22449508124826484, 0.22419530846213745, 0.22389553567600992, 0.22359576288988237, 0.22329599010375467, 0.22299621731762703, 0.22269644453149945, 0.2223966717453718, 0.2220972457084607, 0.2218004351792078, 0.22150362464995488, 0.22120681412070187, 0.2209100035914488, 0.2206131930621959, 0.22031638253294303, 0.22002041873403744, 0.21972831613942503, 0.21943621354481282, 0.21914546828375595, 0.21885702283750155, 0.21857034613928508, 0.21828366944106883, 0.2179979516850024, 0.21771405229939622, 0.2174301529137899, 0.2171472202995229, 0.2168673291223431, 0.21658894814895158, 0.21631056717556002, 0.21603218620216846, 0.21575447727034314, 0.21547774468752212, 0.21520276727910218, 0.2149320202458516, 0.21466298473072415, 0.21439512249093615, 0.21413235552738458, 0.2138715391491546, 0.2136107227709251, 0.2133499063926954, 0.21309075021461915, 0.21283528827601994, 0.2125838117720995, 0.21233688229948025, 0.21209084619196636, 0.21184834184912807, 0.21160583750628992, 0.2113633331634517, 0.21112082882061337, 0.21087844210904702, 0.21063845429777467, 0.21039846648650234, 0.21015847867522977, 0.2099184908639573, 0.20967850305268504, 0.2094385152414124, 0.20919852743013997, 0.2089585396188676, 0.20871855180759535, 0.20847856399632278, 0.2082385761850504, 0.20799858837377802, 0.2077586005625056, 0.20751861275123312, 0.20727862493996072, 0.20703863712868809, 0.20680193409672895, 0.20657000242909282, 0.20633807076145674, 0.20610613909382056, 0.2058742074261846, 0.2056422757585484, 0.20541034409091227, 0.20517968578379267, 0.20495202304966, 0.20472436031552738, 0.20449669758139466, 0.20426903484726197, 0.2040413721131293, 0.20381370937899673, 0.20358619050986204, 0.20336175064769146, 0.20313950209337403, 0.20291748232183476, 0.20270106550365935, 0.20248694997188857, 0.20227283444011762, 0.20205871890834687, 0.201844603376576, 0.20163048784480517, 0.20141746539192767, 0.20120563974957692, 0.2009938141072262, 0.20078198846487558, 0.20057016282252513, 0.2003583371801745, 0.20014651153782384, 0.19993468589547322, 0.19972286025312255, 0.1995125162034825, 0.19930420518059278, 0.19909589415770282, 0.19888758313481314, 0.19867927211192316, 0.19847096108903337, 0.19826376304220977, 0.19805816325815953, 0.1978525634741096, 0.1976479028469825, 0.19744453105516457, 0.19724115926334673, 0.1970377874715286, 0.1968344156797107, 0.1966310438878929, 0.19642767209607487, 0.19622430030425694, 0.19602092851243905, 0.19581755672062112, 0.19561418492880323, 0.1954108131369854, 0.19520744134516738, 0.19500406955334948, 0.19480069776153153, 0.19459732596971352, 0.19439395417789554, 0.19419058238607775, 0.1939872252962926, 0.19378394834291665, 0.19358067138954083, 0.19337739443616495, 0.19317411748278907, 0.1929708405294132, 0.19276756357603728, 0.1925642866226616, 0.19236100966928557, 0.19215773271590983, 0.19195445576253392, 0.19175194854016422, 0.19155126284219273, 0.19135057714422118, 0.19114989144624953, 0.1909492057482779, 0.1907485200503062, 0.19054783435233455, 0.19034714865436317, 0.19014646295639143, 0.18994614579900895, 0.18974776860292814, 0.18954939140684715, 0.18935101421076636, 0.18915263701468552, 0.1889542598186048, 0.18875588262252402, 0.18855757626631134, 0.18836095397815983, 0.18816433169000815, 0.18796770940185664, 0.18777164610571861, 0.18757678300246158, 0.1873819198992049, 0.18718705679594796, 0.18699219369269113, 0.18679733058943418, 0.1866024674861774, 0.1864076043829205, 0.18621274127966367, 0.18601899998876106, 0.18582693451258153, 0.18563486903640233, 0.18544280356022316, 0.18525073808404388, 0.18505867260786463, 0.1848686604112746, 0.18468254297454265, 0.1844983258495525, 0.18431479206661697, 0.1841326852035097, 0.18395261916863442, 0.18377255313375898, 0.18359248709888365, 0.18341242106400837, 0.18323235502913288, 0.18305278133830996, 0.18287575618563087, 0.1827015622500054, 0.18252736831438, 0.18235317437875434, 0.18217898044312905, 0.1820047865075036, 0.18183059257187822, 0.1816576854683348, 0.18148601728309724, 0.1813158278119727, 0.1811477814506901, 0.18097973508940748, 0.18081209105376383, 0.18064639019180806, 0.1804818850779467, 0.18031737996408548, 0.1801531290664119, 0.17999027072623966, 0.1798274123860674, 0.17966455404589518, 0.17950169570572297, 0.17933934469623866, 0.17917747067066084, 0.17901559664508293, 0.17885372261950488, 0.17869184859392687, 0.17852997456834885, 0.178368100542771, 0.17820622651719312, 0.17804435249161507, 0.1778824784660372, 0.17772060444045923, 0.17755873041488127, 0.17739685638930322, 0.17723498236372542, 0.17707310833814757, 0.1769112343125694, 0.17674936028699145, 0.17658748626141355, 0.1764256122358355, 0.17626373821025756, 0.17610186418467985, 0.1759399901591017, 0.17577811613352376, 0.17561624210794588, 0.17545436808236783, 0.1752924940567899, 0.175130620031212, 0.1749687460056339, 0.17480687198005604, 0.17464499795447808, 0.17448312392890022, 0.17432124990332232, 0.1741593758777442, 0.17399750185216634, 0.1738356278265883, 0.1736737538010103, 0.17351305697461955, 0.17335416037513487, 0.17319526377565034, 0.17303636717616558, 0.172877470576681, 0.17271857397719628, 0.17255967737771175, 0.17240078077822701, 0.1722418841787423, 0.17208475364583017, 0.17193083614760502, 0.17177691864938008, 0.171623001151155, 0.17146908365292995, 0.1713151661547049, 0.17116124865647986, 0.1710079749668606, 0.1708555179886823, 0.1707038010029946, 0.17055462938919427, 0.1704063982449873, 0.1702605546496202, 0.1701154713221819, 0.16997157269487584, 0.16982767406756977, 0.16968377544026378, 0.16953987681295765, 0.16939597818565158, 0.16925207955834556, 0.16910818093103944, 0.1689642823037334, 0.1688203836764274, 0.16867658509812314, 0.16853489268588692, 0.1683932002736505, 0.16825150786141416, 0.16810981544917789, 0.16796812303694145, 0.1678264306247051, 0.16768473821246868, 0.1675430458002323, 0.16740135338799603, 0.16725966097575962, 0.16711796856352323, 0.16697627615128693, 0.16683458373905055, 0.16669289132681425, 0.1665511989145779, 0.16640950650234163, 0.16626781409010508, 0.16612673231059039, 0.1659863228858182, 0.16584591346104588, 0.16570550403627382, 0.16556509461150148, 0.1654246851867294, 0.16528505696372095, 0.16514811451350644, 0.1650111720632922, 0.16487422961307777, 0.16473728716286345, 0.1646003447126491, 0.16446340226243453, 0.16432658756650934, 0.16419157907989654, 0.164056570593284, 0.16392156210667122, 0.16378655362005853, 0.16365154513344582, 0.1635167866268276, 0.163383595760685, 0.16325119057949697, 0.16311881546005388, 0.16298715067160569, 0.16285660337843647, 0.16272605608526713, 0.16259550879209803, 0.16246496149892864, 0.16233441420575925, 0.16220386691259, 0.16207331961942073, 0.16194277232625148, 0.1618122250330821, 0.16168167773991274, 0.1615511304467436, 0.16142058315357427, 0.161290035860405, 0.1611594885672358, 0.16102894127406642, 0.16089839398089717, 0.16076784668772784, 0.16063860650837838, 0.16051114678008083, 0.1603836870517832, 0.16025622732348568, 0.1601287675951881, 0.1600013078668906, 0.15987384813859307, 0.1597463884102954, 0.15961979631276946, 0.15949546487691535, 0.15937113344106113, 0.1592473856857266, 0.15912503431377462, 0.15900268294182265, 0.15888033156987064, 0.15875798019791865, 0.15863562882596652, 0.15851327745401436, 0.15839117010497059, 0.15827005941169753, 0.15814894871842464, 0.15802783802515166, 0.15790672733187863, 0.1577856166386056, 0.1576645059453326, 0.15754339525205968, 0.15742228455878682, 0.15730117386551373, 0.15718006317224076, 0.1570589524789679, 0.15693784178569484, 0.1568167310924218, 0.15669562039914894, 0.15657450970587583, 0.156453399012603, 0.1563322883193299, 0.156211177626057, 0.15609006693278385, 0.15596895623951093, 0.15584784554623796, 0.155726734852965, 0.15560562415969206, 0.15548451346641917, 0.1553635586479447, 0.15524362100505587, 0.1551236833621672, 0.15500374571927825, 0.15488380807638955, 0.15476387043350068, 0.1546439327906119, 0.15452399514772314, 0.1544040575048344, 0.15428411986194565, 0.15416419206370616, 0.15404538785445873, 0.15392658364521145, 0.15380777943596405, 0.1536889752267167, 0.15357017101746942, 0.1534513668082221, 0.15333256259897474, 0.1532137583897274, 0.15309495418048003, 0.15297614997123282, 0.15285845040032361, 0.1527425716444929, 0.15262669288866196, 0.15251081413283119, 0.1523949353770004, 0.1522790566211695, 0.15216317786533878, 0.15204729910950798, 0.15193142035367704, 0.15181554159784622, 0.15169966284201547, 0.15158378408618453, 0.15146790533035379, 0.151352026574523, 0.15123614781869207, 0.15112026906286133, 0.1510043903070306, 0.1508887723627998, 0.15077449544067636, 0.15066021851855302, 0.1505459415964297, 0.1504316646743064, 0.150317387752183, 0.15020311083005972, 0.15008883390793637, 0.14997455698581297, 0.14986028006368957, 0.14974600314156636, 0.1496317262194429, 0.14951744929731967, 0.14940317237519624, 0.14928889545307283, 0.14917461853094968, 0.14906034160882628, 0.14894606468670293, 0.14883178776457962, 0.14871810401137708, 0.14860547546078984, 0.1484928469102026, 0.14838021835961535, 0.1482675898090282, 0.14815496125844088, 0.14804233270785364, 0.1479297041572665, 0.14781707560667934, 0.14770444705609198, 0.14759181850550487, 0.1474797633012156, 0.14736900987315113, 0.1472582564450867, 0.1471475030170222, 0.14703674958895777, 0.14692599616089325, 0.1468152427328288, 0.1467044893047645, 0.14659373587670005, 0.1464835476276687, 0.14637473987502284, 0.14626593212237693, 0.1461571243697312, 0.1460483166170852, 0.14593950886443932, 0.1458307011117934, 0.14572189335914754, 0.1456130856065017, 0.1455042778538559, 0.1453954701012099, 0.14528666234856405, 0.14517785459591823, 0.14506904684327235, 0.14496023909062639, 0.14485143133798062, 0.14474262358533474, 0.14463505582815409, 0.1445291257656036, 0.14442319570305315, 0.14431726564050276, 0.1442113355779523, 0.1441054055154018, 0.1439994754528513, 0.14389354539030091, 0.14378761532775045, 0.14368168526519998, 0.14357575520264965, 0.14346982514009912, 0.14336389507754865, 0.14325796501499818, 0.14315203495244772, 0.14304610488989727, 0.1429401748273468, 0.14283424476479634, 0.14272831470224595, 0.14262238463969548, 0.14251645457714499, 0.14241052451459454, 0.14230459445204413, 0.1421987684283742, 0.14209398561976, 0.14198966883941777, 0.14188696271381968, 0.14178425658822147, 0.1416816822059964, 0.141580825166089, 0.14147996812618163, 0.14137911108627416, 0.14127825404636662, 0.14117739700645923, 0.14107653996655184, 0.1409756829266445, 0.14087482588673697, 0.14077396884682955, 0.14067311180692216, 0.14057225476701465, 0.14047139772710732, 0.14037054068719984, 0.14026968364729245, 0.14016882660738508, 0.14006796956747758, 0.13996711252757021, 0.13986625548766277, 0.13976539844775532, 0.13966454140784784, 0.13956368436794048, 0.13946282732803308, 0.13936197028812566, 0.1392611132482182, 0.1391614037514571]
    
    # Create a figure instance
    fig = plt.figure(1, figsize=(9, 6))
    plt.plot(accuracy, label='accuracy')

    plt.legend()
    plt.show()
    fig.savefig('question1aAccuracy.png', bbox_inches='tight')

    # Create a figure instance
    fig = plt.figure(1, figsize=(9, 6))
    plt.plot(forward, label='forward loss')
    # plt.plot(backward, label='backward loss')
    plt.legend()
    plt.show()
    # Save the figure
    fig.savefig('question1aLoss.png', bbox_inches='tight')

def question1b():
    train_features, test_features, train_targets, test_targets = load_data('mnist-binary', fraction=1, base_folder='.')
    learner = GradientDescent('hinge', learning_rate=1e-4)

    iterNum = []
    forward = []
    backward = []
    accur = []

    train_features_with_bias = np.c_[train_features, np.ones((train_features.shape[0], 1))]
    learner.fit(train_features, train_targets, batch_size=1, max_iter=train_features.shape[0])

    # for i in range(1000):
        # learner.fit(train_features, train_targets, batch_size=1, max_iter=train_features.shape[0])

    #     old = learner.loss.forward(train_features_with_bias, learner.model, train_targets)
    #     forward.append(old)
    #     pred = learner.predict(train_features)

    #     w = learner.model - learner.loss.backward(train_features_with_bias, learner.model, train_targets) * learner.learning_rate

    #     new = learner.loss.forward(train_features_with_bias, w, train_targets)

    #     backward.append(new)
    #     iterNum.append(i)

    #     accuracy = 0
    #     for j in range(pred.shape[0]):
    #         accuracy += 1 if pred[j] == train_targets[j] else 0

    #     accuracy /= pred.shape[0]
    #     accur.append(accuracy)

    #     if np.abs(new - old) < 1e-4:
    #         break
    
    # print("iteration: ", iterNum)
    # print("accuracy: ", accur)
    # print("forward loss: ", forward)
    # print("backward loss: ", backward)
    # print(train_features.shape[0])

def visual1b():

    

    iterNum = []
    accuracy = [0.328, 0.328, 0.525, 0.746, 0.746, 0.746, 0.784, 0.783, 0.783, 0.841, 0.841, 0.841, 0.841, 0.841, 0.853, 0.853, 0.865, 0.865, 0.866, 0.876, 0.876, 0.877, 0.876, 0.877, 0.878, 0.877, 0.877, 0.878, 0.877, 0.878, 0.878, 0.898, 0.945, 0.944, 0.945, 0.946, 0.946, 0.945, 0.956, 0.955, 0.956, 0.959, 0.958, 0.958, 0.958, 0.958, 0.958, 0.959, 0.958, 0.961, 0.958, 0.958, 0.957, 0.957, 0.958, 0.957, 0.957, 0.957, 0.961, 0.961, 0.968, 0.968, 0.968, 0.966, 0.966, 0.966, 0.966, 0.966, 0.966, 0.966, 0.965, 0.966, 0.966, 0.966, 0.965, 0.965, 0.965, 0.965, 0.965, 0.965, 0.966, 0.966, 0.965, 0.965, 0.966, 0.965, 0.966, 0.966, 0.966, 0.966, 0.966, 0.969, 0.968, 0.969, 0.969, 0.968, 0.969, 0.969, 0.969, 0.968, 0.967, 0.967, 0.969, 0.969, 0.97, 0.97, 0.97, 0.97, 0.969, 0.97, 0.969, 0.97, 0.97, 0.969, 0.969, 0.97, 0.97, 0.969, 0.97, 0.97, 0.969, 0.969, 0.969, 0.969, 0.969, 0.97, 0.969, 0.969, 0.969, 0.969, 0.97, 0.969, 0.97, 0.969, 0.97, 0.969, 0.969, 0.969, 0.97, 0.978, 0.977, 0.977, 0.977, 0.978, 0.977, 0.978, 0.978, 0.978, 0.977, 0.977, 0.978, 0.978, 0.977, 0.978, 0.977, 0.978, 0.977, 0.978, 0.978, 0.977, 0.978, 0.977, 0.977, 0.978, 0.978, 0.978, 0.977, 0.977, 0.978, 0.978, 0.977, 0.977, 0.977, 0.977, 0.977, 0.978, 0.977, 0.977, 0.978, 0.977, 0.977, 0.978, 0.977, 0.977, 0.977, 0.978, 0.978, 0.978, 0.978, 0.976, 0.976, 0.976, 0.977, 0.976, 0.976, 0.976, 0.976, 0.974, 0.975, 0.974, 0.975, 0.974, 0.975, 0.975, 0.974, 0.974, 0.974, 0.975, 0.974, 0.974, 0.974, 0.974, 0.975, 0.974, 0.974, 0.977, 0.977, 0.976, 0.976, 0.976, 0.976, 0.977, 0.976, 0.976, 0.976, 0.977, 0.976, 0.976, 0.976, 0.976, 0.976, 0.977, 0.976, 0.977, 0.977, 0.976, 0.976, 0.971, 0.97, 0.97, 0.974, 0.974, 0.974, 0.974, 0.974, 0.973, 0.974, 0.974, 0.974, 0.973, 0.973, 0.973, 0.974, 0.974, 0.973, 0.974, 0.974, 0.974, 0.976, 0.976, 0.975, 0.976, 0.976, 0.976, 0.977, 0.977, 0.977, 0.977, 0.977, 0.977, 0.977, 0.976, 0.977, 0.977, 0.976, 0.977, 0.976, 0.976, 0.976, 0.977, 0.977, 0.976, 0.977, 0.977, 0.976, 0.98, 0.979, 0.98, 0.98, 0.98, 0.98, 0.98, 0.979, 0.979, 0.98, 0.98, 0.981, 0.98, 0.981, 0.98, 0.98, 0.98, 0.98, 0.983, 0.982, 0.982, 0.982, 0.983, 0.983, 0.983, 0.982, 0.982, 0.982, 0.982, 0.983, 0.983, 0.982, 0.983, 0.983, 0.982, 0.982, 0.982, 0.982, 0.982, 0.983, 0.982, 0.982, 0.983, 0.983, 0.983, 0.983, 0.982, 0.983, 0.982, 0.982, 0.982, 0.983, 0.982, 0.982, 0.983, 0.983, 0.983, 0.982, 0.982, 0.982, 0.983, 0.982, 0.982, 0.983, 0.983, 0.982, 0.982, 0.982, 0.983, 0.983, 0.983, 0.983, 0.988, 0.988, 0.988, 0.988, 0.988, 0.987, 0.988, 0.988, 0.987, 0.988, 0.988, 0.987, 0.987, 0.988, 0.988, 0.988, 0.988, 0.988, 0.988, 0.988, 0.987, 0.988, 0.987, 0.987, 0.987, 0.987, 0.987, 0.987, 0.987, 0.988, 0.987, 0.988, 0.988, 0.988, 0.987, 0.986, 0.985, 0.985, 0.986, 0.986, 0.986, 0.985, 0.986, 0.985, 0.985, 0.986, 0.986, 0.986, 0.985, 0.986, 0.985, 0.985, 0.985, 0.985, 0.986, 0.985, 0.986, 0.986, 0.985, 0.985, 0.985, 0.985, 0.985, 0.985, 0.986, 0.985, 0.985, 0.986, 0.985, 0.986, 0.986, 0.985, 0.986, 0.985, 0.985, 0.985, 0.985, 0.986, 0.986, 0.986, 0.986, 0.986, 0.985, 0.986, 0.985, 0.985, 0.985, 0.985, 0.986, 0.986, 0.986, 0.986, 0.986, 0.985, 0.986, 0.985, 0.985, 0.985, 0.986, 0.986, 0.985, 0.986, 0.985, 0.985, 0.985, 0.985, 0.985, 0.986, 0.986, 0.985, 0.985, 0.986, 0.985, 0.986, 0.986, 0.986, 0.985, 0.986, 0.985, 0.985, 0.986, 0.986, 0.986, 0.985, 0.986, 0.985, 0.985, 0.986, 0.986, 0.985, 0.985, 0.986, 0.985, 0.985, 0.986, 0.986, 0.986, 0.986, 0.986, 0.985, 0.986, 0.985, 0.986, 0.985, 0.986, 0.986, 0.985, 0.986, 0.986, 0.985, 0.985, 0.986, 0.985, 0.986, 0.985, 0.985, 0.986, 0.988, 0.988, 0.987, 0.988, 0.987, 0.987, 0.988, 0.988, 0.987, 0.988, 0.988, 0.987, 0.987, 0.988, 0.988, 0.988, 0.985, 0.985, 0.985, 0.986, 0.986, 0.986, 0.986, 0.986, 0.986, 0.986, 0.985, 0.986, 0.985, 0.986, 0.985, 0.985, 0.985, 0.986, 0.985, 0.985, 0.986, 0.986, 0.986, 0.986, 0.986, 0.985, 0.985, 0.986, 0.986, 0.985, 0.986, 0.985, 0.986, 0.986, 0.986, 0.985, 0.986, 0.986, 0.986, 0.985, 0.986, 0.986, 0.985, 0.986, 0.985, 0.985, 0.99, 0.989, 0.989, 0.992, 0.992, 0.991, 0.991, 0.991, 0.992, 0.992, 0.992, 0.994, 0.994, 0.993, 0.993, 0.993, 0.994, 0.994, 0.993, 0.994, 0.994, 0.994, 0.993, 0.993, 0.993, 0.993, 0.994, 0.994, 0.994, 0.993, 0.993, 0.993, 0.993, 0.993, 0.993, 0.994, 0.994, 0.994, 0.993, 0.994, 0.993, 0.993, 0.993, 0.994, 0.993, 0.994, 0.994, 0.993, 0.993, 0.993, 0.993, 0.994, 0.993, 0.994, 0.993, 0.993, 0.994, 0.994, 0.994, 0.994, 0.993, 0.994, 0.993, 0.994, 0.994, 0.994, 0.993, 0.993, 0.994, 0.993, 0.994, 0.995, 0.995, 0.994, 0.995, 0.995, 0.994, 0.995, 0.995, 0.994, 0.994, 0.994, 0.995, 0.995, 0.995, 0.994, 0.995, 0.995, 0.995, 0.994, 0.995, 0.994, 0.994, 0.994, 0.994, 0.995, 0.994, 0.995, 0.995, 0.994, 0.994, 0.995, 0.994, 0.994, 0.994, 0.994, 0.995, 0.995, 0.995, 0.994, 0.994, 0.995, 0.995, 0.995, 0.994, 0.994, 0.994, 0.994, 0.995, 0.994, 0.995, 0.995, 0.995, 0.995, 0.995, 0.995, 0.994, 0.995, 0.995, 0.994, 0.994, 0.995, 0.994, 0.994, 0.995, 0.994, 0.995, 0.994, 0.995, 0.995, 0.994, 0.995, 0.995, 0.995, 0.995, 0.995, 0.995, 0.995, 0.995, 0.994, 0.994, 0.994, 0.994, 0.994, 0.995, 0.995, 0.994, 0.995, 0.995, 0.994, 0.995, 0.995, 0.995, 0.995, 0.995, 0.994, 0.994, 0.994, 0.994, 0.995, 0.995, 0.994, 0.995, 0.994, 0.994, 0.995, 0.994, 0.995, 0.995, 0.995, 0.995, 0.995, 0.995, 0.995, 0.995, 0.994, 0.994, 0.995, 0.995, 0.993, 0.992, 0.993, 0.992, 0.993, 0.992, 0.992, 0.992, 0.993, 0.992, 0.992, 0.992, 0.993, 0.993, 0.992, 0.993, 0.993, 0.993, 0.993, 0.992, 0.992, 0.993, 0.993, 0.993, 0.992, 0.992, 0.992, 0.992, 0.993, 0.993, 0.992, 0.992, 0.992, 0.993, 0.992, 0.993, 0.993, 0.993, 0.992, 0.992, 0.994, 0.994, 0.994, 0.993, 0.993, 0.993, 0.994, 0.994, 0.994, 0.994, 0.994, 0.993, 0.994, 0.994, 0.994, 0.993, 0.993, 0.993, 0.994, 0.994, 0.993, 0.994, 0.994, 0.993, 0.994, 0.993, 0.993, 0.993, 0.994, 0.994, 0.993, 0.993, 0.993, 0.994, 0.994, 0.993, 0.994, 0.993, 0.994, 0.994, 0.994, 0.994, 0.993, 0.993, 0.993, 0.994, 0.994, 0.993, 0.994, 0.993, 0.993, 0.994, 0.994, 0.994, 0.994, 0.993, 0.993, 0.994, 0.993, 0.993, 0.993, 0.993, 0.994, 0.993, 0.994, 0.993, 0.993, 0.994, 0.993, 0.993, 0.994, 0.994, 0.993, 0.994, 0.993, 0.994, 0.993, 0.993, 0.993, 0.993, 0.994, 0.993, 0.994, 0.994, 0.994, 0.993, 0.994, 0.993, 0.993, 0.993, 0.994, 0.993, 0.994, 0.994, 0.994, 0.994, 0.994, 0.994, 0.994, 0.994, 0.994, 0.993, 0.994, 0.994, 0.994, 0.993, 0.993, 0.994, 0.994, 0.994, 0.993, 0.994, 0.993, 0.994, 0.993, 0.994, 0.994, 0.994, 0.994, 0.994, 0.993, 0.994, 0.994, 0.993, 0.994, 0.993, 0.993, 0.994, 0.993, 0.994, 0.993, 0.993, 0.993, 0.994, 0.995, 0.995, 0.995, 0.995, 0.994, 0.995, 0.994, 0.994, 0.995, 0.995, 0.994, 0.994, 0.995, 0.995, 0.994, 0.994, 0.994, 0.994, 0.995, 0.994, 0.994, 0.995, 0.994, 0.994, 0.995, 0.994, 0.994, 0.994, 0.995, 0.994, 0.995, 0.995, 0.995, 0.994, 0.995, 0.994, 0.995, 0.995, 0.995, 0.994, 0.994, 0.995, 0.994, 0.994, 0.995, 0.995, 0.994, 0.995, 0.995, 0.995, 0.994, 0.994, 0.995, 0.994, 0.994, 0.995, 0.995, 0.994, 0.995, 0.994, 0.994]

    forward = [1.7445417666496108, 1.7445417666496108, 1.1242693702199589, 0.7754790878132942, 0.7754790878132942, 0.7754790878132942, 0.6464518667841646, 0.6464518667841646, 0.6464518667841646, 0.4193983571844719, 0.4193983571844719, 0.4193983571844719, 0.4193983571844719, 0.4193983571844719, 0.382519532269397, 0.382519532269397, 0.3554382967527918, 0.3554382967527918, 0.3554382967527918, 0.3235240381561024, 0.3235240381561024, 0.3235240381561024, 0.3235240381561024, 0.3235240381561024, 0.3263863855069353, 0.3263863855069353, 0.3263863855069353, 0.3263863855069353, 0.3263863855069353, 0.3263863855069353, 0.3263863855069353, 0.2490569001605161, 0.1468968149045948, 0.1468968149045948, 0.1468968149045948, 0.14154707021966648, 0.14154707021966648, 0.14154707021966648, 0.10936136940281697, 0.10936136940281697, 0.10936136940281697, 0.10424158511246022, 0.10424158511246022, 0.10424158511246022, 0.10424158511246022, 0.10424158511246022, 0.10424158511246022, 0.10424158511246022, 0.10424158511246022, 0.09898276469716485, 0.09785268008232843, 0.09785268008232843, 0.09785268008232843, 0.09785268008232843, 0.09785268008232843, 0.09785268008232843, 0.09785268008232843, 0.09785268008232843, 0.09084479936090918, 0.09084479936090918, 0.08040244111487892, 0.08040244111487892, 0.08040244111487892, 0.08201193400211335, 0.08201193400211335, 0.08201193400211335, 0.08201193400211335, 0.08201193400211335, 0.08201193400211335, 0.08201193400211335, 0.08201193400211335, 0.08201193400211335, 0.08201193400211335, 0.08201193400211335, 0.08201193400211335, 0.08201193400211335, 0.08201193400211335, 0.08201193400211335, 0.08201193400211335, 0.08201193400211335, 0.08201193400211335, 0.08201193400211335, 0.08201193400211335, 0.08201193400211335, 0.08201193400211335, 0.08201193400211335, 0.08201193400211335, 0.08201193400211335, 0.08201193400211335, 0.08201193400211335, 0.08201193400211335, 0.07562409243998232, 0.07562409243998232, 0.07562409243998232, 0.07562409243998232, 0.07562409243998232, 0.07562409243998232, 0.07562409243998232, 0.07562409243998232, 0.07562409243998232, 0.07776242305625883, 0.07776242305625883, 0.06954109621944901, 0.06954109621944901, 0.06954109621944901, 0.06954109621944901, 0.06954109621944901, 0.06954109621944901, 0.06954109621944901, 0.06954109621944901, 0.06954109621944901, 0.06954109621944901, 0.06954109621944901, 0.06954109621944901, 0.06954109621944901, 0.06954109621944901, 0.06954109621944901, 0.06954109621944901, 0.06954109621944901, 0.06954109621944901, 0.06954109621944901, 0.06954109621944901, 0.06954109621944901, 0.06954109621944901, 0.06954109621944901, 0.06954109621944901, 0.06954109621944901, 0.06954109621944901, 0.06954109621944901, 0.06954109621944901, 0.06954109621944901, 0.06954109621944901, 0.06954109621944901, 0.06954109621944901, 0.06954109621944901, 0.06954109621944901, 0.06954109621944901, 0.06954109621944901, 0.06954109621944901, 0.054631387749113976, 0.054631387749113976, 0.054631387749113976, 0.054631387749113976, 0.054631387749113976, 0.054631387749113976, 0.054631387749113976, 0.054631387749113976, 0.054631387749113976, 0.054631387749113976, 0.054631387749113976, 0.054631387749113976, 0.054631387749113976, 0.054631387749113976, 0.054631387749113976, 0.054631387749113976, 0.054631387749113976, 0.054631387749113976, 0.054631387749113976, 0.054631387749113976, 0.054631387749113976, 0.054631387749113976, 0.054631387749113976, 0.054631387749113976, 0.054631387749113976, 0.054631387749113976, 0.054631387749113976, 0.054631387749113976, 0.054631387749113976, 0.054631387749113976, 0.054631387749113976, 0.054631387749113976, 0.054631387749113976, 0.054631387749113976, 0.054631387749113976, 0.054631387749113976, 0.054631387749113976, 0.054631387749113976, 0.054631387749113976, 0.054631387749113976, 0.054631387749113976, 0.054631387749113976, 0.054631387749113976, 0.054631387749113976, 0.054631387749113976, 0.054631387749113976, 0.054631387749113976, 0.054631387749113976, 0.054631387749113976, 0.054631387749113976, 0.054730663666210995, 0.054730663666210995, 0.054730663666210995, 0.054730663666210995, 0.054730663666210995, 0.054730663666210995, 0.054730663666210995, 0.054730663666210995, 0.054687300370469025, 0.054687300370469025, 0.054687300370469025, 0.054687300370469025, 0.054687300370469025, 0.054687300370469025, 0.054687300370469025, 0.054687300370469025, 0.054687300370469025, 0.054687300370469025, 0.054687300370469025, 0.054687300370469025, 0.054687300370469025, 0.054687300370469025, 0.054687300370469025, 0.054687300370469025, 0.054687300370469025, 0.054687300370469025, 0.05080539034261775, 0.05080539034261775, 0.05080539034261775, 0.05080539034261775, 0.05080539034261775, 0.05080539034261775, 0.05080539034261775, 0.05080539034261775, 0.05080539034261775, 0.05080539034261775, 0.05080539034261775, 0.05080539034261775, 0.05080539034261775, 0.05086226840564592, 0.05086226840564592, 0.05086226840564592, 0.05086226840564592, 0.05086226840564592, 0.05086226840564592, 0.05086226840564592, 0.05086226840564592, 0.05086226840564592, 0.07105750681092653, 0.07105750681092653, 0.07105750681092653, 0.06634923166143104, 0.06634923166143104, 0.06634923166143104, 0.06634923166143104, 0.06634923166143104, 0.06634923166143104, 0.06634923166143104, 0.06634923166143104, 0.06634923166143104, 0.06634923166143104, 0.06634923166143104, 0.06634923166143104, 0.06634923166143104, 0.06634923166143104, 0.06634923166143104, 0.06634923166143104, 0.06634923166143104, 0.06634923166143104, 0.0624867347745397, 0.0624867347745397, 0.0624867347745397, 0.06174086009263035, 0.06174086009263035, 0.05488547112038122, 0.05488547112038122, 0.05488547112038122, 0.05488547112038122, 0.05488547112038122, 0.05488547112038122, 0.05488547112038122, 0.05488547112038122, 0.05488547112038122, 0.05488547112038122, 0.05488547112038122, 0.05488547112038122, 0.05488547112038122, 0.05488547112038122, 0.05488547112038122, 0.05488547112038122, 0.05488547112038122, 0.05488547112038122, 0.05488547112038122, 0.05488547112038122, 0.05488547112038122, 0.05488547112038122, 0.04777730549303944, 0.04777730549303944, 0.04777730549303944, 0.04777730549303944, 0.04777730549303944, 0.04777730549303944, 0.04777730549303944, 0.04777730549303944, 0.04777730549303944, 0.04511146409205037, 0.04511146409205037, 0.04511146409205037, 0.04511146409205037, 0.04511146409205037, 0.04511146409205037, 0.04511146409205037, 0.04511146409205037, 0.04511146409205037, 0.037601610314502106, 0.037601610314502106, 0.037601610314502106, 0.037601610314502106, 0.037601610314502106, 0.037601610314502106, 0.037601610314502106, 0.037601610314502106, 0.037601610314502106, 0.037601610314502106, 0.037601610314502106, 0.037601610314502106, 0.037601610314502106, 0.037601610314502106, 0.037601610314502106, 0.037601610314502106, 0.037601610314502106, 0.037601610314502106, 0.037601610314502106, 0.037601610314502106, 0.037601610314502106, 0.037601610314502106, 0.037601610314502106, 0.037601610314502106, 0.037601610314502106, 0.037601610314502106, 0.037601610314502106, 0.03760288726195838, 0.03760288726195838, 0.03760288726195838, 0.03760288726195838, 0.03760288726195838, 0.03760288726195838, 0.03760288726195838, 0.03760288726195838, 0.03760288726195838, 0.03760288726195838, 0.03760288726195838, 0.03760288726195838, 0.03760288726195838, 0.03760288726195838, 0.03760288726195838, 0.03760288726195838, 0.03760288726195838, 0.03760288726195838, 0.03760288726195838, 0.03760288726195838, 0.03760288726195838, 0.03760288726195838, 0.03760288726195838, 0.03760288726195838, 0.03760288726195838, 0.03760288726195838, 0.03760288726195838, 0.02899944636985402, 0.02899944636985402, 0.02899944636985402, 0.02899944636985402, 0.02899944636985402, 0.02899944636985402, 0.02899944636985402, 0.02899944636985402, 0.02899944636985402, 0.02899944636985402, 0.02899944636985402, 0.02899944636985402, 0.02899944636985402, 0.02899944636985402, 0.02899944636985402, 0.02899944636985402, 0.02899944636985402, 0.02899944636985402, 0.02899944636985402, 0.02899944636985402, 0.02899944636985402, 0.02899944636985402, 0.02899944636985402, 0.02899944636985402, 0.02899944636985402, 0.02899944636985402, 0.02899944636985402, 0.02899944636985402, 0.02899944636985402, 0.02899944636985402, 0.02899944636985402, 0.02899944636985402, 0.02899944636985402, 0.02899944636985402, 0.02899944636985402, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.03682056463711074, 0.036598901935996554, 0.036598901935996554, 0.036598901935996554, 0.036598901935996554, 0.036598901935996554, 0.036598901935996554, 0.036598901935996554, 0.036598901935996554, 0.036598901935996554, 0.036598901935996554, 0.036598901935996554, 0.036598901935996554, 0.036598901935996554, 0.036598901935996554, 0.036598901935996554, 0.036598901935996554, 0.036598901935996554, 0.036598901935996554, 0.036598901935996554, 0.036598901935996554, 0.036598901935996554, 0.036598901935996554, 0.033994600234967254, 0.033994600234967254, 0.033994600234967254, 0.033994600234967254, 0.033994600234967254, 0.033994600234967254, 0.033994600234967254, 0.033994600234967254, 0.033994600234967254, 0.033994600234967254, 0.033994600234967254, 0.033994600234967254, 0.033994600234967254, 0.033994600234967254, 0.033994600234967254, 0.033994600234967254, 0.03132234846390172, 0.03132234846390172, 0.03132234846390172, 0.03132234846390172, 0.03132234846390172, 0.03132234846390172, 0.03132234846390172, 0.03132234846390172, 0.03132234846390172, 0.03132234846390172, 0.03132234846390172, 0.03132234846390172, 0.03132234846390172, 0.03132234846390172, 0.03132234846390172, 0.03132234846390172, 0.03132234846390172, 0.03132234846390172, 0.03132234846390172, 0.03132234846390172, 0.03132234846390172, 0.03132234846390172, 0.03132234846390172, 0.03132234846390172, 0.03132234846390172, 0.03132234846390172, 0.03132234846390172, 0.03132234846390172, 0.03132234846390172, 0.03132234846390172, 0.03132234846390172, 0.03132234846390172, 0.03132234846390172, 0.03132234846390172, 0.03132234846390172, 0.03132234846390172, 0.03132234846390172, 0.03132234846390172, 0.03132234846390172, 0.03132234846390172, 0.03132234846390172, 0.03132234846390172, 0.03132234846390172, 0.03132234846390172, 0.03132234846390172, 0.03132234846390172, 0.025647306795700108, 0.025647306795700108, 0.025647306795700108, 0.02065956461526442, 0.02065956461526442, 0.02065956461526442, 0.02065956461526442, 0.02065956461526442, 0.02065956461526442, 0.02065956461526442, 0.02065956461526442, 0.018140192240612668, 0.018140192240612668, 0.018140192240612668, 0.018140192240612668, 0.018140192240612668, 0.018140192240612668, 0.018140192240612668, 0.018140192240612668, 0.018140192240612668, 0.018140192240612668, 0.018140192240612668, 0.018140192240612668, 0.018140192240612668, 0.018140192240612668, 0.018140192240612668, 0.018140192240612668, 0.018140192240612668, 0.018140192240612668, 0.018140192240612668, 0.018140192240612668, 0.018140192240612668, 0.018140192240612668, 0.018140192240612668, 0.018140192240612668, 0.018140192240612668, 0.018140192240612668, 0.018140192240612668, 0.018140192240612668, 0.018140192240612668, 0.018140192240612668, 0.018140192240612668, 0.018140192240612668, 0.018084511985111608, 0.018084511985111608, 0.018084511985111608, 0.018084511985111608, 0.018084511985111608, 0.018084511985111608, 0.018084511985111608, 0.018084511985111608, 0.018084511985111608, 0.018084511985111608, 0.018084511985111608, 0.018084511985111608, 0.018084511985111608, 0.018084511985111608, 0.018084511985111608, 0.018084511985111608, 0.018084511985111608, 0.018084511985111608, 0.018084511985111608, 0.018084511985111608, 0.018084511985111608, 0.018084511985111608, 0.018084511985111608, 0.018084511985111608, 0.018084511985111608, 0.018084511985111608, 0.018084511985111608, 0.018084511985111608, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.017285162272482724, 0.020248888299341714, 0.020248888299341714, 0.020248888299341714, 0.020248888299341714, 0.020248888299341714, 0.020248888299341714, 0.020248888299341714, 0.020248888299341714, 0.020248888299341714, 0.020248888299341714, 0.020248888299341714, 0.020248888299341714, 0.020248888299341714, 0.020248888299341714, 0.020248888299341714, 0.020248888299341714, 0.020248888299341714, 0.020248888299341714, 0.020248888299341714, 0.020248888299341714, 0.020248888299341714, 0.020248888299341714, 0.020248888299341714, 0.020248888299341714, 0.020248888299341714, 0.020248888299341714, 0.020248888299341714, 0.020248888299341714, 0.020248888299341714, 0.020248888299341714, 0.020248888299341714, 0.020248888299341714, 0.020248888299341714, 0.020248888299341714, 0.020248888299341714, 0.020248888299341714, 0.020248888299341714, 0.020248888299341714, 0.020248888299341714, 0.020248888299341714, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.01762081659888148, 0.017033963945130796, 0.017033963945130796, 0.017033963945130796, 0.017033963945130796, 0.017033963945130796, 0.017033963945130796, 0.017033963945130796, 0.017033963945130796, 0.017033963945130796, 0.017033963945130796, 0.017033963945130796, 0.017033963945130796, 0.017033963945130796, 0.017033963945130796, 0.017033963945130796, 0.017033963945130796, 0.017033963945130796, 0.017033963945130796, 0.017033963945130796, 0.017033963945130796, 0.017033963945130796, 0.017033963945130796, 0.017033963945130796, 0.017033963945130796, 0.017033963945130796, 0.017033963945130796, 0.017033963945130796, 0.017033963945130796, 0.017033963945130796, 0.017033963945130796, 0.017033963945130796, 0.017033963945130796, 0.017033963945130796, 0.017033963945130796, 0.017033963945130796, 0.017033963945130796, 0.017033963945130796, 0.017033963945130796, 0.017033963945130796, 0.017033963945130796, 0.017033963945130796, 0.017033963945130796, 0.017033963945130796, 0.017033963945130796, 0.017033963945130796, 0.017033963945130796, 0.017033963945130796, 0.017033963945130796, 0.01666254882535325, 0.01666254882535325, 0.01666254882535325, 0.01666254882535325, 0.01666254882535325, 0.01666254882535325, 0.01666254882535325, 0.01666254882535325, 0.01666254882535325, 0.01666254882535325, 0.01666254882535325, 0.01666254882535325, 0.01666254882535325]

    backward = []
    
    # Create a figure instance
    fig = plt.figure(1, figsize=(9, 6))
    plt.plot(accuracy, label='accuracy')

    plt.legend()
    plt.show()
    fig.savefig('question1bAccuracy.png', bbox_inches='tight')

    # Create a figure instance
    fig = plt.figure(1, figsize=(9, 6))
    plt.plot(forward, label='forward loss')
    # plt.plot(backward, label='backward loss')
    plt.legend()
    plt.show()
    # Save the figure
    fig.savefig('question1bLoss.png', bbox_inches='tight')

def question2a():
    train_features, test_features, train_targets, test_targets = load_data('synthetic', fraction=1, base_folder='.')
    # learner = GradientDescent('hinge', learning_rate=1e-4)
    bias = [0.5, -0.5, -1.5, -2.5, -3.5, -4.5, -5.5]

    w = np.ones((train_features.shape[1]+1))
    loss = ZeroOneLoss()
    forward = []
    one = np.ones((train_features.shape[0],1))
    train_features = np.c_[train_features,one]

    for b in bias:
        w[-1] = b
        forward.append(loss.forward(train_features, w, train_targets))
    
    
    # Create a figure instance
    fig = plt.figure(1, figsize=(9, 6))
    plt.plot(bias, forward, label='forward loss')
    plt.xlabel('Bias')
    plt.ylabel('Loss value')
    # plt.plot(backward, label='backward loss')
    plt.legend()
    plt.show()
    # Save the figure
    fig.savefig('question2aLoss.png', bbox_inches='tight')

def question2b():
    train_features, test_features, train_targets, test_targets = load_data('synthetic', fraction=1, base_folder='.')
    # learner = GradientDescent('hinge', learning_rate=1e-4)
    bias = [0.5, -0.5, -1.5, -2.5, -3.5, -4.5, -5.5]

    w = np.ones((train_features.shape[1]+1))
    loss = ZeroOneLoss()
    forward = []
    one = np.ones((train_features.shape[0],1))
    train_features = np.c_[train_features,one]

    for b in bias:
        w[-1] = b
        forward.append(loss.forward(train_features[-5:-1, :], w, train_targets[:4]))
    
    print(forward)
    # Create a figure instance
    fig = plt.figure(1, figsize=(9, 6))
    plt.plot(bias, forward, label='forward loss')
    plt.xlabel('Bias')
    plt.ylabel('Loss value')
    # plt.plot(backward, label='backward loss')
    plt.legend()
    plt.show()
    # Save the figure
    fig.savefig('question2bLoss.png', bbox_inches='tight')

def question3a():
    train_features, test_features, train_targets, test_targets = load_data('mnist-multiclass', fraction=0.75, base_folder='.')
    learner = MultiClassGradientDescent(loss = 'squared',regularization = 'l1')

    learner.fit(train_features, train_targets, max_iter = 1000)
    print("train_ number:", train_targets.shape[0])

    pred = learner.predict(test_features)
    print(test_targets)
    print(pred)
    matrix = confusion_matrix(test_targets, pred)


    # Create a figure instance
    fig = plt.figure(1, figsize=(9, 6))

    plt.table(cellText=matrix, loc='center')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('predicted class')
    plt.ylabel('ground truth clas')
    plt.legend()
    plt.show()
    # Save the figure
    fig.savefig('question3a.png', bbox_inches='tight')

def question4a():
    l1 = []
    l2 = []
    lambdas = [1e-3, 1e-2, 1e-1, 1, 10, 100]
    train_features, test_features, train_targets, test_targets = load_data('mnist-binary', fraction=1.0, base_folder='.')
    print('label = 1:' + str(np.size(np.where(train_targets == 1))))
    print('label = -1:' + str(np.size(np.where(train_targets == -1))))
    for l in lambdas:
        print("lambdas: ", l)
        learner1 = GradientDescent('squared', regularization = 'l1', learning_rate=1e-5, reg_param = l)
        # learner1.fit(train_features, train_targets, batch_size = 10, max_iter = 2000)
        learner1.fit(train_features, train_targets, batch_size = 50, max_iter = 2000)
        w1 = np.abs(learner1.model)
        l1.append(np.size(np.where(w1 > 0.001)))

        learner2 = GradientDescent('squared', regularization = 'l2', learning_rate=1e-5, reg_param = l)
        # learner2.fit(train_features, train_targets, batch_size = 10, max_iter = 2000)
        learner2.fit(train_features, train_targets, batch_size = 50, max_iter = 2000)
        w2 = np.abs(learner2.model)
        l2.append(np.size(np.where(w2 > 0.001)))

    # Create a figure instance
    fig = plt.figure(1, figsize=(9, 6))

    print("l1: ", l1)
    print("l2: ", l2)

    plt.plot(lambdas,l1,label='l1')
    plt.plot(lambdas,l2,label='l2')
    plt.xlabel('lambda')
    plt.ylabel('# of non-zero model weights')
    plt.legend()
    plt.show()
    # Save the figure
    fig.savefig('question4a.png', bbox_inches='tight')

def question4c():
    train_features, test_features, train_targets, test_targets = load_data('mnist-binary', fraction=1.0, base_folder='.')
    learner = GradientDescent('squared', regularization = 'l1', learning_rate=1e-5, reg_param = 1)
    learner.fit(train_features, train_targets, batch_size = 50, max_iter = 2000)
    weight = np.abs(learner.model[:-1].reshape(28,28))
    weight[np.where(weight > 0.001)] = 1
    weight[np.where(weight < 0.001)] = 0

    # Create a figure instance
    fig = plt.figure(1, figsize=(9, 6))

    fig,ax = plt.subplots(figsize=(10,20))
    im = ax.imshow(weight)
    fig.colorbar(im, pad = 0.03)
    ax.axis('off')
    # plt.show()

    # Save the figure
    fig.savefig('question4c.png', bbox_inches='tight')

    # Create a figure instance
    # nsameples, nx, ny = train_features.shape
    fig = plt.figure(1, figsize=(9, 6))
    tmp = train_features[0].reshape(28, 28)
    plt.imshow(np.array(tmp), cmap=plt.get_cmap('gray'))
    # plt.plot(weight)
    plt.legend()
    # plt.show()
    # Save the figure
    fig.savefig('question4c_digit.png', bbox_inches='tight')

    for i in range(10):
        # Create a figure instance
        # nsameples, nx, ny = train_features.shape
        fig = plt.figure(1, figsize=(9, 6))
        tmp = train_features[i].reshape(28, 28)
        plt.imshow(np.array(tmp), cmap=plt.get_cmap('gray'))
        # plt.plot(weight)
        plt.legend()
        # plt.show()
        # Save the figure
        fig.savefig('question4c_digit' + str(i) + '.png', bbox_inches='tight')

if __name__ == "__main__":
    
    start = time.time()
    print('Starting example experiment')

    # question1a()
    # visual1a()
    # question1b()
    # visual1b()
    # question2a()
    # question2b()
    # question3a()
    # question4a()
    question4c()
    
    
    
    
    
    
    
    
    
    
    
    
    
    print('Finished example experiment')
    end = time.time()

    print("sepnd time: ", end - start)