import torch
import torchvision
import torchvision.transforms as transforms
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import pandas as pd
from matplotlib.pyplot import imread
import numpy as np
import os
import struct
from array import array as pyarray
import random
import torch.utils.data as utils
import matplotlib.pyplot as plt
# import torch.utils.data as Data
os.environ['KMP_DUPLICATE_LIB_OK']='True'

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


class DogsDataset:

    def __init__(self, path_to_dogsset, num_classes=10):
        """
        This is a class that loads DogSet into memory. Give it the path
        to the DogSet folder on your machine and it will load it when you
        initialize this object.

        Training examples are stored in `self.trainX` and `self.trainY`
        for the images and labels, respectively. Validation examples
        are similarly in `self.validX` and `self.validY`, and test
        examples are in `self.testX` and self.testY`.

        You can also access the training, validation, and testing sets
        with the `get_training_examples()`, `get_validation_exmples()`
        and `get_test_examples()` functions.

        The `trainX`, `validX`, and `testX` arrays are of shape:
            `[num_examples, height, width, n_channels]`

        (For DogSet `height` == `width`.)

        The `trainY`, `validY`, and `testY` arrays are of shape:
            `[num_examples]`

        """
        self.path_to_dogs_csv = os.path.join(path_to_dogsset, 'dogs.csv')
        self.images_dir = os.path.join(path_to_dogsset, 'images')
        np.random.seed(0)
        self.num_classes = num_classes
        # Load in all the data we need from disk
        self.metadata = pd.read_csv(self.path_to_dogs_csv)
        self.semantic_labels = dict(zip(
            self.metadata['numeric_label'],
            self.metadata['semantic_label']
        ))

        self.trainX, self.trainY = self._load_data('train')
        self.validX, self.validY = self._load_data('valid')
        self.testX, self.testY = self._load_data('test')
        self.all_index = np.arange(len(self.trainX) + len(self.testX))
        self.all_count = 0
        self.valid_count = 0

    def get_train_examples(self):
        """
        Gets all training examples in DogSet.
        :return:
            (np.ndarray, np.ndarray), all training examples and all training labels
        """
        return self.trainX, self.trainY

    def get_validation_examples(self):
        """
        Gets all validation examples in DogSet.
        :return:
            (np.ndarray, np.ndarray), all validation examples and all validation labels
        """
        return self.validX, self.validY

    def get_test_examples(self):
        """
        Gets all test examples in DogSet.
        :return:
            (np.ndarray, np.ndarray), all test examples and all test labels
        """
        return self.testX, self.testY

    def get_examples_by_label(self, partition, label, num_examples=None):
        """
        Returns the entire subset of the partition that belongs to the class
        specified by label. If num_examples is None, returns all relevant
        examples.
        """
        if partition == 'train':
            X = self.trainX[self.trainY == label]
        elif partition == 'valid':
            X = self.validX[self.validY == label]
        elif partition == 'test':
            X = self.testX[self.testY == label]
        else:
            raise ValueError('Partition {} does not exist'.format(partition))
        return X if num_examples == None else X[:num_examples]

    def get_semantic_label(self, numeric_label):
        """
        Returns the string representation of the numeric class label (e.g.,
        the numberic label 1 maps to the semantic label 'miniature_poodle').
        """
        return self.semantic_labels[numeric_label]

    def _load_data(self, partition='train'):
        """
        Loads a single data partition from file.
        """
        print("loading %s..." % partition)
        Y = None
        if partition == 'all':
            X = self._get_images(
                self.metadata[~self.metadata.partition.isin(['train', 'valid', 'test'])])
            X = self._preprocess(X, False)
            return X
        else:
            X, Y = self._get_images_and_labels(
                self.metadata[self.metadata.partition == partition])
            X = self._preprocess(X, True)
            return X, Y

    def _get_images_and_labels(self, df):
        """
        Fetches the data based on image filenames specified in df.
        If training is true, also loads the labels.
        """
        X, y = [], []
        for i, row in df.iterrows():
            label = row['numeric_label']
            if label >= self.num_classes: continue
            image = imread(os.path.join(self.images_dir, row['filename']))
            X.append(image)
            y.append(row['numeric_label'])
        return np.array(X), np.array(y).astype(int)

    def _get_images(self, df):
        X = []
        for i, row in df.iterrows():
            image = imread(os.path.join(self.images_dir, row['filename']))
            X.append(image)
        return np.array(X)

    def _preprocess(self, X, is_train):
        """
        Preprocesses the data partition X by normalizing the images
        """
        X = self._normalize(X, is_train)
        return X

    def _normalize(self, X, is_train):
        """
        Normalizes the partition to have mean 0 and variance 1. Learns the
        mean and standard deviation parameters from the training set and
        applies these values when normalizing the other data partitions.

        Returns:
            the normalized data as a numpy array.
        """
        if is_train:
            self.image_mean = np.mean(X, axis=(0,1,2))
            self.image_std = np.std(X, axis=(0,1,2))
        return (X - self.image_mean) / self.image_std

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.conv1 = nn.Conv2d(3, 6, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(64*64*3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        

    def forward(self, x):
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # return F.log_softmax(x)
        return x
start = time.time()

dog = DogsDataset(path_to_dogsset='DogSet')
(train_f, train_l) = (dog.trainX, dog.trainY)
(test_f, test_l) = (dog.testX, dog.testY)
(valid_f, valid_l) = (dog.validX, dog.validY)
# counter = 0
# for i in train_l:
#     counter += 1
#     # print("train label: ", i)

# print("train total: ", counter)
# counter = 0
# for i in test_l:
#     counter += 1
#     # print("train label: ", i)

# print("test total: ", counter)
# counter = 0
# for i in valid_l:
#     counter += 1
#     # print("train label: ", i)

# print("valid total: ", counter)

# print("trainf: ", train_f.shape)

# print("sema: ", len(dog.semantic_labels))
# print(dog.semantic_labels)
# end = time.time()
# print("finish: ", end - start)
# exit(1)

# 
# n_epochs = 100
# batch_size_train = 10
# batch_size_test = 1000
# learning_rate = 0.01
# momentum = 0.5
# log_interval = 10

# random_seed = 1
# torch.backends.cudnn.enabled = False
# torch.manual_seed(random_seed)
# 
# network = Net()
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
# digits_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# data_set_size = 500
data_size = [500]

#train
# for epoch in range(2):  # loop over the dataset multiple times
# def train(epoch):
#     running_loss = 0.0
#     for i, data in enumerate(my_dataloader, 0):
#         # get the inputs; data is a list of [inputs, labels]
#         inputs, labels = data
#         print("")
#         print("label")
#         print(labels)
#         # zero the parameter gradients
#         optimizer.zero_grad()

#         # forward + backward + optimize
#         outputs = network(inputs)
#         print("output")
#         print(outputs)
#         print("label")
#         print(labels)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         # print statistics
#         running_loss += loss.item()
#         if i % 2000 == 1999:    # print every 2000 mini-batches
#             print('[%d, %5d] loss: %.3f' %
#                   (epoch + 1, i + 1, running_loss / 2000))
#             running_loss = 0.0

#     print('Finished Training')

# def test():
#     network.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             output = network(data)
#             test_loss += F.nll_loss(output, target, size_average=False).item()
#             pred = output.data.max(1, keepdim=True)[1]
#             correct += pred.eq(target.data.view_as(pred)).sum()
#     test_loss /= len(test_loader.dataset)
#     test_losses.append(test_loss)
#     print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))

# start = time.time()
# read data
print("read train data")
# train_features, test_features, train_targets, test_targets = load_mnist_data(100, fraction=1.0, examples_per_class=500, mnist_folder='.')

# tmp_train_fe = []
# tmp_train_tar = []
# counter = 0
# for i in range(train_features.shape[0]):
#     if digits_count[train_targets[i]] <= data_set_size/10:
#         tmp_train_fe.append(train_features[i])
#         tmp_train_tar.append(train_targets[i])
#         counter += 1
#         digits_count[train_targets[i]] += 1
#     if counter >= data_set_size:
#         break
# tmp_train_fe = np.array(tmp_train_fe)
# tmp_train_fe = torch.from_numpy(tmp_train_fe)
# tmp_train_tar = np.array(tmp_train_tar)
# tmp_train_tar = torch.from_numpy(tmp_train_tar)
print("done get data set")
# exit(1)



print("read test data")
# test_loader = torch.utils.data.DataLoader(
#                             torchvision.datasets.MNIST('/files/', train=False, download=True,
#                                                         transform=torchvision.transforms.Compose([
#                                                         torchvision.transforms.ToTensor(),
#                                                         torchvision.transforms.Normalize(
#                                                             (0.1307,), (0.3081,))
#                                                         ])),
#                             batch_size=batch_size_test, shuffle=True)
test_features, test_targets = dog.testX, dog.testY
test_features = test_features.reshape(-1,64*64*3)
test_features = torch.from_numpy(test_features)
test_batch_size = len(test_targets)
test_targets = torch.from_numpy(test_targets)
test_targets = test_targets.long()
test_dataset = utils.TensorDataset(test_features,test_targets)
test_loader = utils.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

# val_features, val_targets = dog.validX, dog.validY
# val_features = val_features.reshape(-1,64*64*3)
# val_features = torch.from_numpy(val_features)
# val_batch_size = len(val_targets)
# val_targets = torch.from_numpy(val_targets)
# val_targets = val_targets.long()
# val_dataset = utils.TensorDataset(val_features,val_targets)
# val_loader = utils.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
validation_x, validation_y = dog.get_validation_examples()
num_validation = validation_x.shape[0]
validation_x = validation_x.reshape(len(validation_x),-1)
validation_x = torch.from_numpy(validation_x)
validation_x = validation_x.reshape(-1,64*64*3)
validation_y = torch.from_numpy(validation_y).long()
validation_dataset = utils.TensorDataset(validation_x,validation_y)
val_loader = utils.DataLoader(validation_dataset,batch_size=32,shuffle=False)

# initialize the net
print("init network")
# network = Net()
# optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)


#training
train_losses = []
val_losses = []
train_counter = []
test_losses = []
test_counter = [i * 500 for i in range(100 + 1)]
time_with_epoch = []
accuracy = []
# times = []
time_with_data_size = []
val_accuracy = []

#
for i in data_size:
    # print("start data size: ", i)
    net = Net()
    net = net.float()
    criteration = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(),lr=0.001)

    nexample, nx, ny, nz = dog.trainX.shape
    tmp = dog.trainX
    tmp = tmp.reshape(nexample, nx*ny*nz)

    train_features1, train_targets1 = tmp, dog.trainY
    train_features1 = torch.from_numpy(train_features1).float()
    train_targets1 = torch.from_numpy(train_targets1).float()
    train_targets1 = train_targets1.long()
    train_dataset1 = utils.TensorDataset(train_features1,train_targets1)
    train_loader1 = utils.DataLoader(train_dataset1, batch_size=32, shuffle=True)
    # train_loader1 = utils.DataLoader(dog, batch_size=32, shuffle=True)

    # test()
    start_this_data_set = time.time()
    terminate_e = 0
    old = 0
    new = 0
    for epoch in range(0, 100):
        terminate_e = epoch
        start_this_epoch = time.time()
        # train_f = []
        # train_t = []
        # rand = random.sample(range(0, len(tmp_train_tar)), batch_size_train)
        # for i in rand:
        #     train_f.append(tmp_train_fe[i])
        #     train_t.append(tmp_train_tar[i])
        # train_f = np.array(train_f)
        # train_t = [np.array([i]) for i in (train_t)]
        # train_t = [[i] for i in (train_t)]
        # train_t = np.array(i) for i in (train_t)
        # print(train_t)
        # tensor_x = torch.stack([torch.Tensor(i) for i in train_f])
        # tensor_y = torch.stack([torch.LongTensor(i) for i in train_t])
        # tensor_y = torch.stack([torch.LongTensor(i) for i in train_t])
        # print(tensor_y)
        # my_dataset = utils.TensorDataset(tensor_x,tensor_y)
        # my_dataloader = utils.DataLoader(my_dataset)

        print("enter epoch: ", epoch)
        # train(epoch)
        # digits_count = [0 * 10]
        # test()
        old = new
        new = 0
        runing_loss = 0.0
        for i, data in enumerate(train_loader1,0):
            inputs, labels = data
            # print("data")
            # print(labels)
            optimizer.zero_grad()
            outputs = net(inputs)
            
            loss = criteration(outputs, labels)
            # old = loss.item()
            loss.backward()
            
            optimizer.step()
            new += loss.item()
            # if new - old < 1e-4:
            #     print("new - old < 1e-4: ", new - old)
                
            #     break
        train_losses.append(new/7665)
        
        vali_epoch = 0
        with torch.no_grad():
            for data in val_loader:
                images, labels = data
                outputs = net(images.float())
                tmp_val_loss = nn.CrossEntropyLoss()
                # loss = criteration(outputs, labels)
                tmp = tmp_val_loss(outputs, labels)
                vali_epoch += tmp.item()
                # _, predicted = torch.max(outputs.data, 1)
                # total += labels.size(0)
                # correct += (predicted == labels).sum().item()

        print("train_epoch: ", new/7665)
        print("vali_epoch: ", vali_epoch/2000)
        val_losses.append(vali_epoch/2000)

        correct = 0
        total = 0
        with torch.no_grad():
            for data in val_loader:
                images, labels = data
                outputs = net(images.float())
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_accuracy.append(correct / total)

        # print("finish epoch: ", epoch)    
        time_with_epoch.append(time.time()-start_this_epoch)
        if abs(new - old) < 1e-4:
                print("new - old < 1e-4: ", abs(new - old))
                print("new: ", new)
                print("old: ", old)
                
                break
        #validation




        
    
    print("terminate epoch: ", terminate_e)
    time_with_data_size.append(time.time() - start_this_data_set)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images.float())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy.append(correct / total)

print("train_losses")
print(train_losses)
print("train_counter")
print(train_counter)
print("test_losses")
print(test_losses)
print("test_counter")
print(test_counter)
print("valid losses")
print(val_losses)
print("time with epoch")
print(time_with_epoch)
print("valid accuracy:")
print(val_accuracy)
print("accuracy")
print(accuracy)

end = time.time()
print("sepnd time: ", end - start)

# fig = plt.figure(1, figsize=(9, 6))

# plt.plot(data_size, time_with_data_size)
# plt.xlabel('examples')
# plt.ylabel('time')
# plt.show()

# # Save the figure
# fig.savefig('questiondog_time.png', bbox_inches='tight')

fig = plt.figure(1, figsize=(9, 6))

num_of_epoch = []
for i in range(100):
    num_of_epoch.append(i)

plt.plot(num_of_epoch, val_losses, label='valid losses')
plt.plot(num_of_epoch, train_losses, label='train losses')
plt.xlabel('epoch')
plt.ylabel('losses')
plt.legend()
plt.show()

# Save the figure
fig.savefig('questiondog_losses.png', bbox_inches='tight')

fig = plt.figure(1, figsize=(9, 6))

plt.plot(num_of_epoch, val_accuracy, label='valid accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.show()

# Save the figure
fig.savefig('questiondog_avl_accur.png', bbox_inches='tight')

# if __name__ == "__main__":
#     start = time.time()
#     question4()



#     end = time.time()
#     print("sepnd time: ", end - start)