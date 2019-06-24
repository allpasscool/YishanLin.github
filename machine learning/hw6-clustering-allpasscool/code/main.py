from mnist import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import numpy as np
import statistics 
from scipy.stats import mannwhitneyu
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from metrics import adjusted_mutual_info_score

fewest = 892

def my_metrics(true_labels, pre_labels):
    correct = 0
    for i in range(10):
        index = np.where(pre_labels == i)
        cluster = true_labels[index]
        label = np.argmax(np.bincount(cluster))
        true = np.where(cluster == label)
        correct += np.size(true)
    return correct / np.size(true_labels)

def  read_training_and_testing():
    path = ''
    # images, labels = load_mnist(dataset="training", path=path)

    # print(images.shape[0])
    # print(labels.shape[0])
    # print(type(images))
    # print(type(labels))
    j = [0] * 10
    sum = 0
    # print("training")
    # for i in labels:
    #     j[i] += 1

    # for i in j:
    #     #sum += i
    #     print(i)
    images, labels = load_mnist(dataset="testing", path=path)
    print("testing")
    for i in labels:
        j[i] += 1

    num = 0
    for i in j:
        sum += i
        print(num)
        num += 1
        print(i)
    print("sum")
    print(sum)

def approach_without_labels_kmeans():
    start = time.time()
    path = ''
    images, labels = load_mnist(dataset="testing", path=path)
    numbers = [0] * 10
    new_samples = []
    new_labels = []
    tmp = 0

    p = np.random.permutation(len(labels))
    images = images[p]
    labels = labels[p]
    
    for j in range(len(labels)):
        # print(tmp)
        if numbers[labels[j]] < fewest:
            tmp += 1
            numbers[labels[j]] += 1
            new_samples.append(images[j])
            new_labels.append(labels[j])
        if tmp == fewest * 10:
            break
    # p = np.random.permutation(fewest)
    # shuffle_images = images[p]
    # shuffle_labels = labels[p]
    # nsameples, nx, ny = shuffle_images.shape
    # d2_train_image = shuffle_images.reshape((nsameples, nx*ny))
    
    new_samples = np.array(new_samples)
    new_labels = np.array(new_labels)

    nsameples, nx, ny = new_samples.shape
    d2_train_image = new_samples.reshape((nsameples, nx*ny))
    
    kmeans = KMeans(n_clusters=10).fit(d2_train_image)
    print(kmeans.labels_)
    print(adjusted_mutual_info_score(kmeans.labels_, new_labels))
    print(len(new_labels))
    print(numbers)

    end = time.time()
    train_time = end - start
    print("spending time")
    print(train_time)

def approach_without_labels_GMM():
    start = time.time()
    path = ''
    images, labels = load_mnist(dataset="testing", path=path)
    numbers = [0] * 10
    new_samples = []
    new_labels = []
    tmp = 0

    p = np.random.permutation(len(labels))
    images = images[p]
    labels = labels[p]
    
    for j in range(len(labels)):
        # print(tmp)
        if numbers[labels[j]] < fewest:
            tmp += 1
            numbers[labels[j]] += 1
            new_samples.append(images[j])
            new_labels.append(labels[j])
        if tmp == fewest * 10:
            break

    # p = np.random.permutation(fewest)
    # shuffle_images = images[p]
    # shuffle_labels = labels[p]
    # nsameples, nx, ny = shuffle_images.shape
    # d2_train_image = shuffle_images.reshape((nsameples, nx*ny))

    new_samples = np.array(new_samples)
    new_labels = np.array(new_labels)

    nsameples, nx, ny = new_samples.shape
    d2_train_image = new_samples.reshape((nsameples, nx*ny))
    
    gmm = GaussianMixture(n_components=10, covariance_type='spherical').fit(d2_train_image)
    output = gmm.predict(d2_train_image)
    print(output)
    print(adjusted_mutual_info_score(output, new_labels))
    print(len(new_labels))
    print(numbers)

    end = time.time()
    train_time = end - start
    print("spending time")
    print(train_time)

def approach_with_labels_kmeans():
    start = time.time()
    path = ''
    images, labels = load_mnist(dataset="testing", path=path)
    numbers = [0] * 10
    new_samples = []
    new_labels = []
    tmp = 0

    p = np.random.permutation(len(labels))
    images = images[p]
    labels = labels[p]
    
    for j in range(len(labels)):
        # print(tmp)
        if numbers[labels[j]] < fewest:
            tmp += 1
            numbers[labels[j]] += 1
            new_samples.append(images[j])
            new_labels.append(labels[j])
        if tmp == fewest * 10:
            break
    # p = np.random.permutation(fewest)
    # shuffle_images = images[p]
    # shuffle_labels = labels[p]
    # nsameples, nx, ny = shuffle_images.shape
    # d2_train_image = shuffle_images.reshape((nsameples, nx*ny))

    new_samples = np.array(new_samples)
    new_labels = np.array(new_labels)

    nsameples, nx, ny = new_samples.shape
    d2_train_image = new_samples.reshape((nsameples, nx*ny))
    
    kmeans = KMeans(n_clusters=10).fit(d2_train_image)
    print(kmeans.labels_)
    sum = 0
    for i in range(fewest*10):
        # print(i)
        if kmeans.labels_[i] == new_labels[i]:
            sum += 1
    # sum1 = 0
    # for i in range(fewest):
    #     tmp = kmeans.predict(shuffle_images[i].reshape(1, nx*ny))
    #     if tmp == new_labels[i]:
    #         sum1 += 1
    print(sum)
    print(sum/fewest/10)
    print(adjusted_mutual_info_score(kmeans.labels_, new_labels) )
    print("my metric")
    print(my_metrics(kmeans.labels_, new_labels))
    # print(sum1)
    # print(sum1/fewest)
    print(len(new_labels))
    print(numbers)
    end = time.time()
    train_time = end - start
    print("spending time")
    print(train_time)

def approach_with_labels_GMM():
    start = time.time()
    path = ''
    images, labels = load_mnist(dataset="testing", path=path)
    numbers = [0] * 10
    new_samples = []
    new_labels = []
    tmp = 0

    p = np.random.permutation(len(labels))
    images = images[p]
    labels = labels[p]
    
    for j in range(len(labels)):
        # print(tmp)
        if numbers[labels[j]] < fewest:
            tmp += 1
            numbers[labels[j]] += 1
            new_samples.append(images[j])
            new_labels.append(labels[j])
        if tmp == fewest * 10:
            break

    # p = np.random.permutation(fewest)
    # shuffle_images = images[p]
    # shuffle_labels = labels[p]
    # nsameples, nx, ny = shuffle_images.shape
    # d2_train_image = shuffle_images.reshape((nsameples, nx*ny))

    new_samples = np.array(new_samples)
    new_labels = np.array(new_labels)

    nsameples, nx, ny = new_samples.shape
    d2_train_image = new_samples.reshape((nsameples, nx*ny))
    
    gmm = GaussianMixture(n_components=10, covariance_type='spherical').fit(d2_train_image)
    output = gmm.predict(d2_train_image)
    print(output)
    sum = 0
    for i in range(fewest*10):
        if output[i] == new_labels[i]:
            sum += 1
    sum1 = 0
    # for i in range(fewest):
    #     tmp = gmm.predict(shuffle_images[i].reshape(1, nx*ny))
    #     if tmp == new_labels[i]:
    #         sum1 += 1
    print(sum)
    print(sum/fewest/10)
    print("my metric")
    print(my_metrics(output, new_labels))
    # print(sum1)
    # print(sum1/fewest)
    print(len(new_labels))
    print(numbers)
    end = time.time()
    train_time = end - start
    print("spending time")
    print(train_time)

def visual_kmeans_1():
    path = ''
    images, labels = load_mnist(dataset="testing", path=path)
    p = np.random.permutation(fewest)
    shuffle_images = images[p]
    shuffle_labels = labels[p]
    nsameples, nx, ny = shuffle_images.shape
    d2_train_image = shuffle_images.reshape((nsameples, nx*ny))
    
    kmeans = KMeans(n_clusters=10).fit(d2_train_image)
    new_samples = [[]] * 10
    num = [0] * 10

    for i in range(fewest):
        new_samples[kmeans.labels_[i]].append(shuffle_images[i])
        num[kmeans.labels_[i]] += 1
    # means = np.array(means)
    # means = np.mean(means, axis=1)
    means0 = np.array(new_samples[9])
    # print(means0.shape[0])
    # for i in range(30):
    #     # define subplot
    #     plt.plot(330 + 1 + i)
    #     # plot raw pixel data
    #     plt.imshow(means0[i], cmap=plt.get_cmap('gray'))
    #     plt.show()
    
    means0 = np.mean(means0, axis=0)

    # print(means0)

    # Create a figure instance
    fig = plt.figure(1, figsize=(9, 6))

    # plot first few images
    for i in range(1):
        # define subplot
        plt.plot(330 + 1 + i)
        # plot raw pixel data
        plt.imshow(means0, cmap=plt.get_cmap('gray'))
    # show the figure
    # plt.show()

    fig.savefig('visual_kmeans_1_9.png', bbox_inches='tight')

def visual_kmeans_2():
    path = ''
    images, labels = load_mnist(dataset="testing", path=path)
    p = np.random.permutation(fewest)
    shuffle_images = images[p]
    shuffle_labels = labels[p]
    nsameples, nx, ny = shuffle_images.shape
    d2_train_image = shuffle_images.reshape((nsameples, nx*ny))
    
    kmeans = KMeans(n_clusters=10).fit(d2_train_image)
    new_samples = [[]] * 10
    num = [0] * 10

    for i in range(fewest):
        new_samples[kmeans.labels_[i]].append(shuffle_images[i])
        num[kmeans.labels_[i]] += 1
    # means = np.array(means)
    # means = np.mean(means, axis=1)
    means0 = np.array(new_samples[0])
    # print(means0.shape[0])
    # for i in range(30):
    #     # define subplot
    #     plt.plot(330 + 1 + i)
    #     # plot raw pixel data
    #     plt.imshow(means0[i], cmap=plt.get_cmap('gray'))
    #     plt.show()
    
    means0 = np.mean(means0, axis=0)
    print(means0)
    print("means0 above")
    sample_0 = np.array(new_samples[0])
    euc_distance = []
    for i in range(num[0]):
        tmp_dis = np.sum((means0.reshape(1, nx*ny) - sample_0[i].reshape(1, nx*ny)) ** 2, axis = 1)
        print(tmp_dis)
        euc_distance.append(tmp_dis)
        
        
    print("index")
    print(np.argmin(np.array(euc_distance)))
    print(euc_distance)
    print(euc_distance[np.argmin(np.array(euc_distance))])
    mean = sample_0[np.argmin(np.array(euc_distance))]
    # print(means0)

    # Create a figure instance
    fig = plt.figure(1, figsize=(9, 6))

    # plot first few images
    for i in range(1):
        # define subplot
        plt.plot(330 + 1 + i)
        # plot raw pixel data
        plt.imshow(mean, cmap=plt.get_cmap('gray'))
    # show the figure
    plt.show()

    fig.savefig('visual_kmeans_2.png', bbox_inches='tight')

def visual_GMM_1():
    start = time.time()
    path = ''
    images, labels = load_mnist(dataset="testing", path=path)
    numbers = [0] * 10
    new_samples = []
    new_labels = []
    tmp = 0
    num = [0] * 10

    p = np.random.permutation(len(labels))
    images = images[p]
    labels = labels[p]
    
    sameple0 = []
    sameple1 = []
    sameple2 = []
    sameple3 = []
    sameple4 = []
    sameple5 = []
    sameple6 = []
    sameple7 = []
    sameple8 = []
    sameple9 = []

    for j in range(len(labels)):
        # print(tmp)
        if numbers[labels[j]] < fewest:
            tmp += 1
            numbers[labels[j]] += 1
            new_samples.append(images[j])
            new_labels.append(labels[j])
        if tmp == fewest * 10:
            break

    # p = np.random.permutation(fewest)
    # shuffle_images = images[p]
    # shuffle_labels = labels[p]
    # nsameples, nx, ny = shuffle_images.shape
    # d2_train_image = shuffle_images.reshape((nsameples, nx*ny))

    new_samples = np.array(new_samples)
    new_labels = np.array(new_labels)

    nsameples, nx, ny = new_samples.shape
    d2_train_image = new_samples.reshape((nsameples, nx*ny))
    
    gmm = GaussianMixture(n_components=10, covariance_type='spherical').fit(d2_train_image)
    output = gmm.predict(d2_train_image)
    print(output)
    sum = 0
    for i in range(fewest*10):
        if output[i] == new_labels[i]:
            sum += 1


    for j in range(fewest*10):
        num[output[j]] += 1
        if output[j] == 0:
            sameple0.append(images[j])
        elif output[j] == 1:
            sameple1.append(images[j])
        elif output[j] == 2:
            sameple2.append(images[j])
        elif output[j] == 3:
            sameple3.append(images[j])
        elif output[j] == 4:
            sameple4.append(images[j])
        elif output[j] == 5:
            sameple5.append(images[j])
        elif output[j] == 6:
            sameple6.append(images[j])
        elif output[j] == 7:
            sameple7.append(images[j])
        elif output[j] == 8:
            sameple8.append(images[j])
        elif output[j] == 9:
            sameple9.append(images[j])

    sum1 = 0
    # for i in range(fewest):
    #     tmp = gmm.predict(shuffle_images[i].reshape(1, nx*ny))
    #     if tmp == new_labels[i]:
    #         sum1 += 1
    print(sum)
    print(sum/fewest/10)
    # print(sum1)
    # print(sum1/fewest)
    print(len(new_labels))
    print(numbers)
    print("train model")
    print(num)

    means0 = np.array(sameple0)
    means0 = np.mean(means0, axis=0)

    means1 = np.array(sameple1)
    means1 = np.mean(means1, axis=0)

    means2 = np.array(sameple2)
    means2 = np.mean(means2, axis=0)

    means3 = np.array(sameple3)
    means3 = np.mean(means3, axis=0)

    means4 = np.array(sameple4)
    means4 = np.mean(means4, axis=0)

    means5 = np.array(sameple5)
    means5 = np.mean(means5, axis=0)

    means6 = np.array(sameple6)
    means6 = np.mean(means6, axis=0)

    means7 = np.array(sameple7)
    means7 = np.mean(means7, axis=0)

    means8 = np.array(sameple8)
    means8 = np.mean(means8, axis=0)

    means9 = np.array(sameple9)
    means9 = np.mean(means9, axis=0)
    
    # Create a figure instance
    fig = plt.figure(1, figsize=(9, 6))

    # plot first few images
    # for i in range(1):

    # define subplot
    plt.subplot(330 + 1 + 0)
    # plot raw pixel data
    plt.imshow(means0, cmap=plt.get_cmap('gray'))
    # show the figure
    # plt.show()
    # define subplot
    plt.subplot(330 + 1 + 1)
    plt.imshow(means1, cmap=plt.get_cmap('gray'))
    # show the figure
    # plt.show()
    # define subplot
    plt.subplot(330 + 1 + 2)
    plt.imshow(means2, cmap=plt.get_cmap('gray'))
    # show the figure
    # plt.show()
    # define subplot
    plt.subplot(330 + 1 + 3)
    plt.imshow(means3, cmap=plt.get_cmap('gray'))
    # show the figure
    # plt.show()
    # define subplot
    plt.subplot(330 + 1 + 4)
    plt.imshow(means4, cmap=plt.get_cmap('gray'))
    # show the figure
    # plt.show()
    # define subplot
    plt.subplot(330 + 1 + 5)
    plt.imshow(means5, cmap=plt.get_cmap('gray'))
    # show the figure
    # plt.show()
    # define subplot
    plt.subplot(330 + 1 + 6)
    plt.imshow(means6, cmap=plt.get_cmap('gray'))
    # show the figure
    # plt.show()
    # define subplot
    plt.subplot(330 + 1 + 7)
    plt.imshow(means7, cmap=plt.get_cmap('gray'))
    # show the figure
    # plt.show()
    # define subplot
    plt.subplot(330 + 1 + 8)
    plt.imshow(means8, cmap=plt.get_cmap('gray'))
    # show the figure
    plt.show()
    fig.savefig('visual1_GMM_1.png', bbox_inches='tight')

    # define subplot
    fig2 = plt.figure(1, figsize=(9, 6))
    # plt.plot(330 + 1 + 0)
    plt.imshow(means9, cmap=plt.get_cmap('gray'))
    # show the figure
    plt.show()

    fig2.savefig('visual1_GMM_2.png', bbox_inches='tight')




    end = time.time()
    train_time = end - start
    print("spending time")
    print(train_time)

def visual_GMM_2():
    start = time.time()
    path = ''
    images, labels = load_mnist(dataset="testing", path=path)
    numbers = [0] * 10
    new_samples = []
    new_labels = []
    tmp = 0
    num = [0] * 10

    p = np.random.permutation(len(labels))
    images = images[p]
    labels = labels[p]
    
    sameple0 = []
    sameple1 = []
    sameple2 = []
    sameple3 = []
    sameple4 = []
    sameple5 = []
    sameple6 = []
    sameple7 = []
    sameple8 = []
    sameple9 = []

    for j in range(len(labels)):
        # print(tmp)
        if numbers[labels[j]] < fewest:
            tmp += 1
            numbers[labels[j]] += 1
            new_samples.append(images[j])
            new_labels.append(labels[j])
        if tmp == fewest * 10:
            break

    # p = np.random.permutation(fewest)
    # shuffle_images = images[p]
    # shuffle_labels = labels[p]
    # nsameples, nx, ny = shuffle_images.shape
    # d2_train_image = shuffle_images.reshape((nsameples, nx*ny))

    new_samples = np.array(new_samples)
    new_labels = np.array(new_labels)

    nsameples, nx, ny = new_samples.shape
    d2_train_image = new_samples.reshape((nsameples, nx*ny))
    
    gmm = GaussianMixture(n_components=10, covariance_type='spherical').fit(d2_train_image)
    output = gmm.predict(d2_train_image)
    print(output)
    sum = 0
    for i in range(fewest*10):
        if output[i] == new_labels[i]:
            sum += 1


    for j in range(fewest*10):
        num[output[j]] += 1
        if output[j] == 0:
            sameple0.append(images[j])
        elif output[j] == 1:
            sameple1.append(images[j])
        elif output[j] == 2:
            sameple2.append(images[j])
        elif output[j] == 3:
            sameple3.append(images[j])
        elif output[j] == 4:
            sameple4.append(images[j])
        elif output[j] == 5:
            sameple5.append(images[j])
        elif output[j] == 6:
            sameple6.append(images[j])
        elif output[j] == 7:
            sameple7.append(images[j])
        elif output[j] == 8:
            sameple8.append(images[j])
        elif output[j] == 9:
            sameple9.append(images[j])

    sum1 = 0
    # for i in range(fewest):
    #     tmp = gmm.predict(shuffle_images[i].reshape(1, nx*ny))
    #     if tmp == new_labels[i]:
    #         sum1 += 1
    print(sum)
    print(sum/fewest/10)
    # print(sum1)
    # print(sum1/fewest)
    print(len(new_labels))
    print(numbers)
    print("train model")
    print(num)

    means0 = np.array(sameple0)
    means0 = np.mean(means0, axis=0)
    sameple0 = np.array(sameple0)

    euc_distance = []
    for i in range(num[0]):
        tmp_dis = np.sum((means0.reshape(1, nx*ny) - sameple0[i].reshape(1, nx*ny)) ** 2, axis = 1)
        # print(tmp_dis)
        euc_distance.append(tmp_dis)
    mean0 = sameple0[np.argmin(np.array(euc_distance))]

    means1 = np.array(sameple1)
    means1 = np.mean(means1, axis=0)
    sameple1 = np.array(sameple1)

    euc_distance = []
    for i in range(num[1]):
        tmp_dis = np.sum((means1.reshape(1, nx*ny) - sameple1[i].reshape(1, nx*ny)) ** 2, axis = 1)
        # print(tmp_dis)
        euc_distance.append(tmp_dis)
    mean1 = sameple1[np.argmin(np.array(euc_distance))]

    means2 = np.array(sameple2)
    means2 = np.mean(means2, axis=0)
    sameple2 = np.array(sameple2)

    euc_distance = []
    for i in range(num[2]):
        tmp_dis = np.sum((means2.reshape(1, nx*ny) - sameple2[i].reshape(1, nx*ny)) ** 2, axis = 1)
        # print(tmp_dis)
        euc_distance.append(tmp_dis)
    mean2 = sameple2[np.argmin(np.array(euc_distance))]

    means3 = np.array(sameple3)
    means3 = np.mean(means3, axis=0)
    sameple3 = np.array(sameple3)

    euc_distance = []
    for i in range(num[3]):
        tmp_dis = np.sum((means3.reshape(1, nx*ny) - sameple3[i].reshape(1, nx*ny)) ** 2, axis = 1)
        # print(tmp_dis)
        euc_distance.append(tmp_dis)
    mean3 = sameple3[np.argmin(np.array(euc_distance))]

    means4 = np.array(sameple4)
    means4 = np.mean(means4, axis=0)
    sameple4 = np.array(sameple4)

    euc_distance = []
    for i in range(num[4]):
        tmp_dis = np.sum((means4.reshape(1, nx*ny) - sameple4[i].reshape(1, nx*ny)) ** 2, axis = 1)
        # print(tmp_dis)
        euc_distance.append(tmp_dis)
    mean4 = sameple4[np.argmin(np.array(euc_distance))]

    means5 = np.array(sameple5)
    means5 = np.mean(means5, axis=0)
    sameple5 = np.array(sameple5)

    euc_distance = []
    for i in range(num[5]):
        tmp_dis = np.sum((means5.reshape(1, nx*ny) - sameple5[i].reshape(1, nx*ny)) ** 2, axis = 1)
        # print(tmp_dis)
        euc_distance.append(tmp_dis)
    mean5 = sameple5[np.argmin(np.array(euc_distance))]

    means6 = np.array(sameple6)
    means6 = np.mean(means6, axis=0)
    sameple6 = np.array(sameple6)

    euc_distance = []
    for i in range(num[6]):
        tmp_dis = np.sum((means6.reshape(1, nx*ny) - sameple6[i].reshape(1, nx*ny)) ** 2, axis = 1)
        # print(tmp_dis)
        euc_distance.append(tmp_dis)
    mean6 = sameple6[np.argmin(np.array(euc_distance))]

    means7 = np.array(sameple7)
    means7 = np.mean(means7, axis=0)
    sameple7 = np.array(sameple7)

    euc_distance = []
    for i in range(num[7]):
        tmp_dis = np.sum((means7.reshape(1, nx*ny) - sameple7[i].reshape(1, nx*ny)) ** 2, axis = 1)
        # print(tmp_dis)
        euc_distance.append(tmp_dis)
    mean7 = sameple7[np.argmin(np.array(euc_distance))]

    means8 = np.array(sameple8)
    means8 = np.mean(means8, axis=0)
    sameple8 = np.array(sameple8)

    euc_distance = []
    for i in range(num[8]):
        tmp_dis = np.sum((means8.reshape(1, nx*ny) - sameple8[i].reshape(1, nx*ny)) ** 2, axis = 1)
        # print(tmp_dis)
        euc_distance.append(tmp_dis)
    mean8 = sameple8[np.argmin(np.array(euc_distance))]

    means9 = np.array(sameple9)
    means9 = np.mean(means9, axis=0)
    sameple9 = np.array(sameple9)

    euc_distance = []
    for i in range(num[9]):
        tmp_dis = np.sum((means9.reshape(1, nx*ny) - sameple9[i].reshape(1, nx*ny)) ** 2, axis = 1)
        # print(tmp_dis)
        euc_distance.append(tmp_dis)
    mean9 = sameple9[np.argmin(np.array(euc_distance))]

    # Create a figure instance
    fig = plt.figure(1, figsize=(9, 6))

    # plot first few images
    # for i in range(1):

    # define subplot
    plt.subplot(330 + 1 + 0)
    # plot raw pixel data
    plt.imshow(means0, cmap=plt.get_cmap('gray'))
    # show the figure
    # plt.show()
    # define subplot
    plt.subplot(330 + 1 + 1)
    plt.imshow(means1, cmap=plt.get_cmap('gray'))
    # show the figure
    # plt.show()
    # define subplot
    plt.subplot(330 + 1 + 2)
    plt.imshow(means2, cmap=plt.get_cmap('gray'))
    # show the figure
    # plt.show()
    # define subplot
    plt.subplot(330 + 1 + 3)
    plt.imshow(means3, cmap=plt.get_cmap('gray'))
    # show the figure
    # plt.show()
    # define subplot
    plt.subplot(330 + 1 + 4)
    plt.imshow(means4, cmap=plt.get_cmap('gray'))
    # show the figure
    # plt.show()
    # define subplot
    plt.subplot(330 + 1 + 5)
    plt.imshow(means5, cmap=plt.get_cmap('gray'))
    # show the figure
    # plt.show()
    # define subplot
    plt.subplot(330 + 1 + 6)
    plt.imshow(means6, cmap=plt.get_cmap('gray'))
    # show the figure
    # plt.show()
    # define subplot
    plt.subplot(330 + 1 + 7)
    plt.imshow(means7, cmap=plt.get_cmap('gray'))
    # show the figure
    # plt.show()
    # define subplot
    plt.subplot(330 + 1 + 8)
    plt.imshow(means8, cmap=plt.get_cmap('gray'))
    # show the figure
    plt.show()
    fig.savefig('visual2_GMM_1.png', bbox_inches='tight')

    # define subplot
    fig2 = plt.figure(1, figsize=(9, 6))
    # plt.plot(330 + 1 + 0)
    plt.imshow(means9, cmap=plt.get_cmap('gray'))
    # show the figure
    plt.show()

    fig2.savefig('visual2_GMM_2.png', bbox_inches='tight')




    end = time.time()
    train_time = end - start
    print("spending time")
    print(train_time)

def handwritting():
    start = time.time()
    path = ''
    images, labels = load_mnist(dataset="testing", path=path)
    numbers = [0] * 10
    new_samples = []
    new_labels = []
    tmp = 0

    p = np.random.permutation(len(labels))
    images = images[p]
    labels = labels[p]
    
    for j in range(len(labels)):
        # print(tmp)
        if numbers[5] < fewest and labels[j] == 5:
            tmp += 1
            numbers[labels[5]] += 1
            new_samples.append(images[j])
            new_labels.append(labels[j])
        if tmp == fewest:
            break

    # p = np.random.permutation(fewest)
    # shuffle_images = images[p]
    # shuffle_labels = labels[p]
    # nsameples, nx, ny = shuffle_images.shape
    # d2_train_image = shuffle_images.reshape((nsameples, nx*ny))

    new_samples = np.array(new_samples)
    new_labels = np.array(new_labels)

    nsameples, nx, ny = new_samples.shape
    d2_train_image = new_samples.reshape((nsameples, nx*ny))
    
    gmm1 = GaussianMixture(n_components=1, covariance_type='full').fit(d2_train_image)
    gmm4 = GaussianMixture(n_components=4, covariance_type='full').fit(d2_train_image)
    gmm10 = GaussianMixture(n_components=10, covariance_type='full').fit(d2_train_image)
    gmm20 = GaussianMixture(n_components=20, covariance_type='full').fit(d2_train_image)
    
    # Create a figure instance
    fig = plt.figure(1, figsize=(9, 6))
    for i in range(5):
        # define subplot
        plt.subplot(330 + 1 + i)
        # plot raw pixel data
        tmp, tmp1 = gmm1.sample()
        # tmp = np.array(tmp).reshape(1, nx*ny)
        tmp = tmp[0].reshape(nx, ny)
        plt.imshow(np.array(tmp), cmap=plt.get_cmap('gray'))
        # print(tmp)
        # print(tmp[0].reshape(nx, ny).shape)
    # show the figure
    print(new_samples[0].shape)

    plt.show()
    fig.savefig('handwritting1.png', bbox_inches='tight')

    # Create a figure instance
    fig = plt.figure(1, figsize=(9, 6))
    for i in range(5):
        # define subplot
        plt.subplot(330 + 1 + i)
        # plot raw pixel data
        tmp, tmp1 = gmm4.sample()
        # tmp = np.array(tmp).reshape(1, nx*ny)
        tmp = tmp[0].reshape(nx, ny)
        plt.imshow(np.array(tmp), cmap=plt.get_cmap('gray'))
        # print(tmp)
        # print(tmp[0].reshape(nx, ny).shape)
    # show the figure
    print(new_samples[0].shape)

    plt.show()
    fig.savefig('handwritting4.png', bbox_inches='tight')

    # Create a figure instance
    fig = plt.figure(1, figsize=(9, 6))
    for i in range(5):
        # define subplot
        plt.subplot(330 + 1 + i)
        # plot raw pixel data
        tmp, tmp1 = gmm10.sample()
        # tmp = np.array(tmp).reshape(1, nx*ny)
        tmp = tmp[0].reshape(nx, ny)
        plt.imshow(np.array(tmp), cmap=plt.get_cmap('gray'))
        # print(tmp)
        # print(tmp[0].reshape(nx, ny).shape)
    # show the figure
    print(new_samples[0].shape)

    plt.show()
    fig.savefig('handwritting10.png', bbox_inches='tight')

    # Create a figure instance
    fig = plt.figure(1, figsize=(9, 6))
    for i in range(5):
        # define subplot
        plt.subplot(330 + 1 + i)
        # plot raw pixel data
        tmp, tmp1 = gmm20.sample()
        # tmp = np.array(tmp).reshape(1, nx*ny)
        tmp = tmp[0].reshape(nx, ny)
        plt.imshow(np.array(tmp), cmap=plt.get_cmap('gray'))
        # print(tmp)
        # print(tmp[0].reshape(nx, ny).shape)
    # show the figure
    print(new_samples[0].shape)

    plt.show()
    fig.savefig('handwritting20.png', bbox_inches='tight')

    print(numbers)
    end = time.time()
    train_time = end - start
    print("spending time")
    print(train_time)


if __name__ == "__main__":
    # read_training_and_testing()
    # approach_without_labels_kmeans()
    # approach_without_labels_GMM()
    # approach_with_labels_kmeans()
    approach_with_labels_GMM()
    # visual_kmeans_1()
    # visual_kmeans_2()
    # visual_GMM_1()
    # visual_GMM_2()
    # handwritting()