import numpy as np

class Tree():
    def __init__(self, value=None, attribute_name="root", attribute_index=None, branches=None):
        """
        This class implements a tree structure with multiple branches at each node.
        If self.branches is an empty list, this is a leaf node and what is contained in
        self.value is the predicted class.

        The defaults for this are for a root node in the tree.

        Arguments:
            branches (list): List of Tree classes. Used to traverse the tree. In a
                binary decision tree, the length of this list is either 2 (for left and
                right branches) or 0 (at a leaf node).
            attribute_name (str): Contains name of attribute that the tree splits the data
                on. Used for visualization (see `DecisionTree.visualize`).
            attribute_index (float): Contains the  index of the feature vector for the
                given attribute. Should match with self.attribute_name.
            value (number): Contains the value that data should be compared to along the
                given attribute.
        """
        self.branches = [] if branches is None else branches
        self.attribute_name = attribute_name
        self.attribute_index = attribute_index
        self.value = value

class DecisionTree():
    def __init__(self, attribute_names):
        """
        TODO: Implement this class.

        This class implements a binary decision tree learner for examples with
        categorical attributes. Use the ID3 algorithm for implementing the Decision
        Tree: https://en.wikipedia.org/wiki/ID3_algorithm

        A decision tree is a machine learning model that fits data with a tree
        structure. Each branching point along the tree marks a decision (e.g.
        today is sunny or today is not sunny). Data is filtered by the value of
        each attribute to the next level of the tree. At the next level, the process
        starts again with the remaining attributes, recursing on the filtered data.

        Which attributes to split on at each point in the tree are decided by the
        information gain of a specific attribute.

        Here, you will implement a binary decision tree that uses the ID3 algorithm.
        Your decision tree will be contained in `self.tree`, which consists of
        nested Tree classes (see above).

        Args:
            attribute_names (list): list of strings containing the attribute names for
                each feature (e.g. chocolatey, good_grades, etc.)

        """
        self.attribute_names = attribute_names
        self.tree = None
        return

    def _check_input(self, features):
        if features.shape[1] != len(self.attribute_names):
            raise ValueError(
                "Number of features and number of attribute names must match!"
            )

    def fit(self, features, targets):
        """
        Takes in the features as a numpy array and fits a decision tree to the targets.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
            targets (np.array): numpy array containing class labels for each of the N
                examples.
        """
        self._check_input(features)
        #all_equal = True
        #for i in range(1, len(targets)):
        #    if targets[i] != targets[i-1]:
        #        all_equal = False
        #        break
        #if all_equal:
        #    self.tree.value = targets[0]
        #    return

        #get best attribute
        #maxE = 0
        #maxIdx = 0
        #for i in range(0, features.shape(1)):
        #    currentE = information_gain(features, i, targets)
        #    if maxE < currentE:
        #        maxE = currentE
        #        maxIdx = i
        #bestAttrIdx = maxIdx
        #Tree(bestAttrIdx)

        #return

        self.tree = self.id3(features, targets, np.arange(features.shape[1]), np.argmax(np.bincount(targets)))

    def id3(self, sample_case, targets, attributes, default):
        #sample_case is empty
        if  len(sample_case) <= 0 or len(sample_case[0]) <= 0:
            return Tree(value=default)
        #all have the same classification
        if np.unique(targets).size == 1:
            return Tree(value=targets[0])
        #attributes is empty
        if attributes.size == 0:
            return Tree(value=np.argmax(np.bincount(targets)))

        #choose the attribute with max information gain
        maxE = 0
        maxIdx = 0
        for i in range(0, attributes.size):
            currentE = information_gain(sample_case, i, targets)
            if maxE <= currentE:
                maxE = currentE
                maxIdx = i
        bestAttrIdx = maxIdx
        bestAttr = attributes[bestAttrIdx]
        #print("C")
        #building subtree
        branch0 = []
        branch1 = []
        subtarget0 = []
        subtarget1 = []
        for i in range(0, targets.size):
            #feature == 0
            if sample_case[i][bestAttrIdx] == 0:
                branch0.append(sample_case[i])
                subtarget0.append(targets[i])
            else:
                branch1.append(sample_case[i])
                subtarget1.append(targets[i])
        branch0 = np.array(branch0)
        branch1 = np.array(branch1)
        if branch0.size:
            #branch0 = np.delete(branch0, [bestAttrIdx])
            branch0 = np.hstack((branch0[:, :bestAttrIdx], branch0[:, bestAttrIdx + 1:]))
        if branch1.size:
            #branch1 = np.delete(branch1, [bestAttrIdx])
            branch1 = np.hstack((branch1[:, :bestAttrIdx], branch1[:, bestAttrIdx + 1:]))

        subtarget0 = np.array(subtarget0)
        subtarget1 = np.array(subtarget1)
        subtree = self.id3(branch0, subtarget0, np.delete(attributes,(bestAttrIdx)), np.argmax(np.bincount(targets)))
        branches = []
        branches.append(subtree)
        subtree = self.id3(branch1, subtarget1, np.delete(attributes, (bestAttrIdx)), np.argmax(np.bincount(targets)))
        branches.append(subtree)

        return Tree(attribute_name=self.attribute_names[bestAttrIdx], attribute_index=bestAttrIdx, branches=branches)

    def predict(self, features):
        """
        Takes in features as a numpy array and predicts classes for each point using
        the trained model.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
        """
        self._check_input(features)

        #tmp = []
        #for i in features:
        #    tmp.append(self.traverse_prediction_tree(i, self.tree))
        #    return np.array(tmp)
        return np.array([self.traverse_prediction_tree(i, self.tree) for i in features])

    def traverse_prediction_tree(self, sample_case, current):
        if current.value is not None:
            return current.value

        if sample_case[current.attribute_index] == 0:
            return self.traverse_prediction_tree(sample_case, current.branches[0])
        else:
            return self.traverse_prediction_tree(sample_case, current.branches[1])

    def _visualize_helper(self, tree, level):
        """
        Helper function for visualize a decision tree at a given level of recursion.
        """
        tab_level = "  " * level
        val = tree.value if tree.value is not None else 0
        print("%d: %s%s == %f" % (level, tab_level, tree.attribute_name, val))

    def visualize(self, branch=None, level=0):
        """
        Visualization of a decision tree. Implemented for you to check your work and to
        use as an example of how to use the given classes to implement your decision
        tree.
        """
        if not branch:
            branch = self.tree
        self._visualize_helper(branch, level)

        for branch in branch.branches:
            self.visualize(branch, level+1)

def information_gain(features, attribute_index, targets):
    """
    TODO: Implement me!

    Information gain is how a decision tree makes decisions on how to create
    split points in the tree. Information gain is measured in terms of entropy.
    The goal of a decision tree is to decrease entropy at each split point as much as
    possible. This function should work perfectly or your decision tree will not work
    properly.

    Information gain is a central concept in many machine learning algorithms. In
    decision trees, it captures how effective splitting the tree on a specific attribute
    will be for the goal of classifying the training data correctly. Consider
    data points S and an attribute A. S is split into two data points given binary A:

        S(A == 0) and S(A == 1)

    Together, the two subsets make up S. If A was an attribute perfectly correlated with
    the class of each data point in S, then all points in a given subset will have the
    same class. Clearly, in this case, we want something that captures that A is a good
    attribute to use in the decision tree. This something is information gain. Formally:

        IG(S,A) = H(S) - H(S|A)

    where H is information entropy. Recall that entropy captures how orderly or chaotic
    a system is. A system that is very chaotic will evenly distribute probabilities to
    all outcomes (e.g. 50% chance of class 0, 50% chance of class 1). Machine learning
    algorithms work to decrease entropy, as that is the only way to make predictions
    that are accurate on testing data. Formally, H is defined as:

        H(S) = sum_{c in (classes in S)} -p(c) * log_2 p(c)

    To elaborate: for each class in S, you compute its prior probability p(c):

        (# of elements of class c in S) / (total # of elements in S)

    Then you compute the term for this class:

        -p(c) * log_2 p(c)

    Then compute the sum across all classes. The final number is the entropy. To gain
    more intution about entropy, consider the following - what does H(S) = 0 tell you
    about S?

    Information gain is an extension of entropy. The equation for information gain
    involves comparing the entropy of the set and the entropy of the set when conditioned
    on selecting for a single attribute (e.g. S(A == 0)).

    For more details: https://en.wikipedia.org/wiki/ID3_algorithm#The_ID3_metrics

    Args:
        features (np.array): numpy array containing features for each example.
        attribute_index (int): which column of features to take when computing the
            information gain
        targets (np.array): numpy array containing labels corresponding to each example.

    Output:
        information_gain (float): information gain if the features were split on the
            attribute_index.
    """

    #To elaborate: for each class in S, you compute its prior probability p(c):
    # (#of elements of class c in S) / (total # of elements in S)
    count1 = 0
    count0 = 0
    len_S = 0
    for i in range(0, len(targets)):
        if targets[i] == 1:
            count1 += 1
        if targets[i] == 0:
            count0 += 1
    S_p_c0 = count1 / len(targets)
    S_p_c1 = 1 - S_p_c0
    # c = np.argmax(np.bincount(features[attribute_index]))

    E_S = -S_p_c0 * np.log2(S_p_c0) - S_p_c1 * np.log2(S_p_c1)
    len_S = count0 + count1

    count1 = 0
    count0 = 0
    for i in range(0, len(targets)):
        if features[i, attribute_index] == 1 and targets[i] == 1:
            count1 += 1
        if features[i, attribute_index] == 1 and targets[i] == 0:
            count0 += 1

    h_p_c0 = 0
    h_p_c1 = 0
    E_h = 0
    if count0 + count1 > 0:
        h_p_c0 = count0 / (count0 + count1)
        h_p_c1 = count1 / (count0 + count1)
        E_h = -h_p_c0 * np.log2(h_p_c0) - h_p_c1 * np.log2(h_p_c1)
        E_h *= (count0 + count1) / len_S

    count1 = 0
    count0 = 0
    for i in range(0, len(targets)):
        if features[i, attribute_index] == 1 and targets[i] == 1:
            count1 += 1
        if features[i, attribute_index] == 1 and targets[i] == 0:
            count0 += 1

    l_p_c0 = 0
    l_p_c1 = 0
    E_l = 0

    if count0 + count1 > 0:
        l_p_c0 = count0 / (count0 + count1)
        l_p_c1 = count1 / (count0 + count1)
        E_l = -l_p_c0 * np.log2(l_p_c0) - l_p_c1 * np.log2(l_p_c1)
        E_l *= (count0 + count1) / len_S



    gain = 0.0
    gain = E_S - E_h - E_l

    return gain

if __name__ == '__main__':
    # construct a fake tree
    attribute_names = ['larry', 'curly', 'moe']
    decision_tree = DecisionTree(attribute_names=attribute_names)
    while len(attribute_names) > 0:
        attribute_name = attribute_names[0]
        if not decision_tree.tree:
            decision_tree.tree = Tree(
                attribute_name=attribute_name,
                attribute_index=decision_tree.attribute_names.index(attribute_name),
                value=0,
                branches=[]
            )
        else:
            decision_tree.tree.branches.append(
                Tree(
                    attribute_name=attribute_name,
                    attribute_index=decision_tree.attribute_names.index(attribute_name),
                    value=0,
                    branches=[]
                )
            )
        attribute_names.remove(attribute_name)
    decision_tree.visualize()
