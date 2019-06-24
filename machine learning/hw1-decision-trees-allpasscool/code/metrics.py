import numpy as np

def confusion_matrix(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the confusion matrix. The confusion 
    matrix for a binary classifier would be a 2x2 matrix as follows:

    [
        [true_negatives, false_positives],
        [false_negatives, true_positives]
    ]

    YOU DO NOT NEED TO IMPLEMENT CONFUSION MATRICES THAT ARE FOR MORE THAN TWO 
    CLASSES (binary).
    
    Compute and return the confusion matrix.

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        confusion_matrix (np.array): 2x2 confusion matrix between predicted and actual labels

    """

    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    confuse_matrix = np.random.uniform(0, 10, size=(2, 2))
    # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    TP = np.sum(np.logical_and(predictions == 1, actual == 1))

    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = np.sum(np.logical_and(predictions == 0, actual == 0))

    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = np.sum(np.logical_and(predictions == 1, actual == 0))

    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN = np.sum(np.logical_and(predictions == 0, actual == 1))

    confuse_matrix[0, 0] = TN
    confuse_matrix[0, 1] = FP
    confuse_matrix[1, 0] = FN
    confuse_matrix[1, 1] = TP

    return confuse_matrix

def accuracy(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the accuracy:

    Hint: implement and use the confusion_matrix function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        accuracy (float): accuracy
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    confuse_matrix = confusion_matrix(actual, predictions)
    #accuracy = (TP + TN) / all
    accur = (confuse_matrix[0,0] + confuse_matrix[1,1]) / (confuse_matrix[0,0] + confuse_matrix[0,1] +
                                                           confuse_matrix[1,0] + confuse_matrix[1,1])

    return accur

def precision_and_recall(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the precision and recall:

    https://en.wikipedia.org/wiki/Precision_and_recall

    Hint: implement and use the confusion_matrix function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        precision (float): precision
        recall (float): recall
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    confuse_matrix = confusion_matrix(actual, predictions)

    #precision = TP / (TP+FP)
    pre = confuse_matrix[1, 1] / (confuse_matrix[1,1] + confuse_matrix[0,1])

    #recall = TP / (TP+FN)
    rec = confuse_matrix[1,1] / (confuse_matrix[1,1] + confuse_matrix[1,0])

    return pre, rec

def f1_measure(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the F1-measure:

   https://en.wikipedia.org/wiki/Precision_and_recall#F-measure

    Hint: implement and use the precision_and_recall function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        f1_measure (float): F1 measure of dataset (harmonic mean of precision and 
        recall)
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    confuse_matrix = confusion_matrix(actual, predictions)
    pre, rec = precision_and_recall(actual, predictions)

    f1 = 2 * (pre * rec) / (pre + rec)

    return f1

