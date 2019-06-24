from mnist import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn import svm
import time
import numpy as np
import statistics 
from scipy.stats import mannwhitneyu

def  read_training_and_testing():
    path = ''
    images, labels = load_mnist(dataset="training", path=path)

    print(images.shape[0])
    print(labels.shape[0])
    print(type(images))
    print(type(labels))
    j = [0] * 10
    sum = 0
    print("training")
    for i in labels:
        j[i] += 1

    for i in j:
        #sum += i
        print(i)
    images, labels = load_mnist(dataset="testing", path=path)
    print("testing")
    for i in labels:
        j[i] += 1

    for i in j:
        sum += i
        print(i)
    print("sum")
    print(sum)

def read_0_in_training_and_testing():
    path = ''
    images, labels = load_mnist(dataset="training", path=path)

    j = [0] * 10
    sum = 0
    print("training")
    for i in range(labels.shape[0]):
        if labels[i] == 0:
            print(images[i])
            plt.imshow(images[i])
            plt.show()

    images, labels = load_mnist(dataset="testing", path=path)
    print("testing")
    for i in labels:
        i[i] += 1

    for i in j:
        sum += i
        print(i)
    print("sum")
    print(sum)

def example_5():
    path = ''

    #linear kernel
    #images, labels = load_mnist(dataset="training", path=path, selection=slice(0, 500, 2))
    images, labels = load_mnist(dataset="training", path=path,)
    images_1 = images[0:2, :,:]
    labels_1 = labels[0:2]
    nsameples, nx, ny = images_1.shape
    d2_train_image = images_1.reshape((nsameples, nx*ny))
    #print(images_500)
    #print(labels_500)
    print(images_1.shape)
    print(labels_1.shape)
    #lin_clf = svm.LinearSVC()
    lin_clf = svm.SVC(kernel='linear')
    print(lin_clf.kernel)
    print(lin_clf)
    start = time.time()
    lin_clf.fit(d2_train_image, labels_1) 
    end = time.time()
    train_time = end - start
    print("traing linear with 2")
    print(train_time)

    images_500 = images[0:500, :,:]
    labels_500 = labels[0:500]
    nsameples, nx, ny = images_500.shape
    d2_train_image = images_500.reshape((nsameples, nx*ny))
    #print(images_500)
    #print(labels_500)
    print(images_500.shape)
    print(labels_500.shape)
    #lin_clf = svm.LinearSVC()
    lin_clf = svm.SVC(kernel='linear')
    print(lin_clf.kernel)
    print(lin_clf)
    start = time.time()
    lin_clf.fit(d2_train_image, labels_500) 
    end = time.time()
    train_time = end - start
    print("traing linear with 500")
    print(train_time)

    images_1000 = images[0:1000, :,:]
    labels_1000 = labels[0:1000]
    nsameples, nx, ny = images_1000.shape
    d2_train_image = images_1000.reshape((nsameples, nx*ny))
    #print(images_500)
    #print(labels_500)
    print(images_1000.shape)
    print(labels_1000.shape)
    #lin_clf = svm.LinearSVC()
    lin_clf = svm.SVC(kernel='linear')
    print(lin_clf.kernel)
    print(lin_clf)
    start = time.time()
    lin_clf.fit(d2_train_image, labels_1000) 
    end = time.time()
    train_time = end - start
    print("traing linear with 1000")
    print(train_time)

    images_2000 = images[0:2000, :,:]
    labels_2000 = labels[0:2000]
    nsameples, nx, ny = images_2000.shape
    d2_train_image = images_2000.reshape((nsameples, nx*ny))
    #print(images_500)
    #print(labels_500)
    print(images_2000.shape)
    print(labels_2000.shape)
    #lin_clf = svm.LinearSVC()
    lin_clf = svm.SVC(kernel='linear')
    print(lin_clf.kernel)
    print(lin_clf)
    start = time.time()
    lin_clf.fit(d2_train_image, labels_2000) 
    end = time.time()
    train_time = end - start
    print("traing linear with 2000")
    print(train_time)

    images_4000 = images[0:4000, :,:]
    labels_4000 = labels[0:4000]
    nsameples, nx, ny = images_4000.shape
    d2_train_image = images_4000.reshape((nsameples, nx*ny))
    #print(images_500)
    #print(labels_500)
    print(images_4000.shape)
    print(labels_4000.shape)
    #lin_clf = svm.LinearSVC()
    lin_clf = svm.SVC(kernel='linear')
    print(lin_clf.kernel)
    print(lin_clf)
    start = time.time()
    lin_clf.fit(d2_train_image, labels_4000) 
    end = time.time()
    train_time = end - start
    print("traing linear with 4000")
    print(train_time)

    #polynomial
    images_1 = images[0:2, :,:]
    labels_1 = labels[0:2]
    nsameples, nx, ny = images_1.shape
    d2_train_image = images_1.reshape((nsameples, nx*ny))
    #print(images_500)
    #print(labels_500)
    print(images_1.shape)
    print(labels_1.shape)
    #lin_clf = svm.LinearSVC()
    lin_clf = svm.SVC(kernel='poly')
    print(lin_clf.kernel)
    print(lin_clf)
    start = time.time()
    lin_clf.fit(d2_train_image, labels_1) 
    end = time.time()
    train_time = end - start
    print("traing polynomial with 2")
    print(train_time)

    images_500 = images[0:500, :,:]
    labels_500 = labels[0:500]
    nsameples, nx, ny = images_500.shape
    d2_train_image = images_500.reshape((nsameples, nx*ny))
    #print(images_500)
    #print(labels_500)
    print(images_500.shape)
    print(labels_500.shape)
    #lin_clf = svm.LinearSVC()
    lin_clf = svm.SVC(kernel='poly')
    print(lin_clf.kernel)
    print(lin_clf)
    start = time.time()
    lin_clf.fit(d2_train_image, labels_500) 
    end = time.time()
    train_time = end - start
    print("traing polynomial with 500")
    print(train_time)

    images_1000 = images[0:1000, :,:]
    labels_1000 = labels[0:1000]
    nsameples, nx, ny = images_1000.shape
    d2_train_image = images_1000.reshape((nsameples, nx*ny))
    #print(images_500)
    #print(labels_500)
    print(images_1000.shape)
    print(labels_1000.shape)
    #lin_clf = svm.LinearSVC()
    lin_clf = svm.SVC(kernel='poly')
    print(lin_clf.kernel)
    print(lin_clf)
    start = time.time()
    lin_clf.fit(d2_train_image, labels_1000) 
    end = time.time()
    train_time = end - start
    print("traing polynomial with 1000")
    print(train_time)

    images_2000 = images[0:2000, :,:]
    labels_2000 = labels[0:2000]
    nsameples, nx, ny = images_2000.shape
    d2_train_image = images_2000.reshape((nsameples, nx*ny))
    #print(images_500)
    #print(labels_500)
    print(images_2000.shape)
    print(labels_2000.shape)
    #lin_clf = svm.LinearSVC()
    lin_clf = svm.SVC(kernel='poly')
    print(lin_clf.kernel)
    print(lin_clf)
    start = time.time()
    lin_clf.fit(d2_train_image, labels_2000) 
    end = time.time()
    train_time = end - start
    print("traing polynomial with 2000")
    print(train_time)

    images_4000 = images[0:4000, :,:]
    labels_4000 = labels[0:4000]
    nsameples, nx, ny = images_4000.shape
    d2_train_image = images_4000.reshape((nsameples, nx*ny))
    #print(images_500)
    #print(labels_500)
    print(images_4000.shape)
    print(labels_4000.shape)
    #lin_clf = svm.LinearSVC()
    lin_clf = svm.SVC(kernel='poly')
    print(lin_clf.kernel)
    print(lin_clf)
    start = time.time()
    lin_clf.fit(d2_train_image, labels_4000) 
    end = time.time()
    train_time = end - start
    print("traing linear with 4000")
    print(train_time)

    #RBF
    images_1 = images[0:2, :,:]
    labels_1 = labels[0:2]
    nsameples, nx, ny = images_1.shape
    d2_train_image = images_1.reshape((nsameples, nx*ny))
    #print(images_500)
    #print(labels_500)
    print(images_1.shape)
    print(labels_1.shape)
    #lin_clf = svm.LinearSVC()
    lin_clf = svm.SVC(kernel='rbf')
    print(lin_clf.kernel)
    print(lin_clf)
    start = time.time()
    lin_clf.fit(d2_train_image, labels_1) 
    end = time.time()
    train_time = end - start
    print("traing rbf with 2")
    print(train_time)

    images_500 = images[0:500, :,:]
    labels_500 = labels[0:500]
    nsameples, nx, ny = images_500.shape
    d2_train_image = images_500.reshape((nsameples, nx*ny))
    #print(images_500)
    #print(labels_500)
    print(images_500.shape)
    print(labels_500.shape)
    #lin_clf = svm.LinearSVC()
    lin_clf = svm.SVC(kernel='rbf')
    print(lin_clf.kernel)
    print(lin_clf)
    start = time.time()
    lin_clf.fit(d2_train_image, labels_500) 
    end = time.time()
    train_time = end - start
    print("traing rbf with 500")
    print(train_time)

    images_1000 = images[0:1000, :,:]
    labels_1000 = labels[0:1000]
    nsameples, nx, ny = images_1000.shape
    d2_train_image = images_1000.reshape((nsameples, nx*ny))
    #print(images_500)
    #print(labels_500)
    print(images_1000.shape)
    print(labels_1000.shape)
    #lin_clf = svm.LinearSVC()
    lin_clf = svm.SVC(kernel='rbf')
    print(lin_clf.kernel)
    print(lin_clf)
    start = time.time()
    lin_clf.fit(d2_train_image, labels_1000) 
    end = time.time()
    train_time = end - start
    print("traing rbf with 1000")
    print(train_time)

    images_2000 = images[0:2000, :,:]
    labels_2000 = labels[0:2000]
    nsameples, nx, ny = images_2000.shape
    d2_train_image = images_2000.reshape((nsameples, nx*ny))
    #print(images_500)
    #print(labels_500)
    print(images_2000.shape)
    print(labels_2000.shape)
    #lin_clf = svm.LinearSVC()
    lin_clf = svm.SVC(kernel='rbf')
    print(lin_clf.kernel)
    print(lin_clf)
    start = time.time()
    lin_clf.fit(d2_train_image, labels_2000) 
    end = time.time()
    train_time = end - start
    print("traing rbf with 2000")
    print(train_time)

    images_4000 = images[0:4000, :,:]
    labels_4000 = labels[0:4000]
    nsameples, nx, ny = images_4000.shape
    d2_train_image = images_4000.reshape((nsameples, nx*ny))
    #print(images_500)
    #print(labels_500)
    print(images_4000.shape)
    print(labels_4000.shape)
    #lin_clf = svm.LinearSVC()
    lin_clf = svm.SVC(kernel='rbf')
    print(lin_clf.kernel)
    print(lin_clf)
    start = time.time()
    lin_clf.fit(d2_train_image, labels_4000) 
    end = time.time()
    train_time = end - start
    print("traing linear with 4000")
    print(train_time)


def question_6():
    #x = np.array([2, 500, 1000, 2000])
    #linear
    #y = np.array([0.0009984970092773438, 0.14991450309753418, 0.3987913131713867, 1.1453661918640137])
    x = np.array([500, 1000, 2000, 4000])
    y = np.array([0.14991450309753418, 0.3987913131713867, 1.1453661918640137, 3.3951447010040283])
    z1 = np.polyfit(x, y, 3)
    print(z1)

    #poly
    #y = np.array([0.0019741058349609375, 0.3778085708618164, 1.478153944015503, 6.185482978820801])
    y = np.array([0.3778085708618164, 1.478153944015503, 6.185482978820801, 30.862067699432373])
    z2 = np.polyfit(x, y, 3)
    print(z2)

    #rbf
    #y = np.array([0.0010004043579101562, 0.3038511276245117, 0.9264698028564453, 2.8603618144989014])
    y = np.array([0.3038511276245117, 0.9264698028564453, 2.8603618144989014, 8.837805271148682])
    z3 = np.polyfit(x, y, 3)
    print(z3)

    print(z1[0] * (100000**3) + z1[1] * (100000**2) + z1[2] * (100000**1) + z1[3])
    count = 1
    print_count = 0
    while True:
        t1 = z1[0] * (count**3) + z1[1] * (count**2) + z1[2] * (count**1) + z1[3]
        t2 = z2[0] * (count**3) + z2[1] * (count**2) + z2[2] * (count**1) + z2[3]
        t3 = z3[0] * (count**3) + z3[1] * (count**2) + z3[2] * (count**1) + z3[3]
        #print(t1)
        if(t1 >= 120 or t2 >= 120 or t3 >= 120):
            print("{} {} {}".format(t1, t2, t3))
            print(count)
            print_count+= 1
            break
        if(t1 > 123 and t2 > 123 and t3 > 123):
            break
        if(count >= 6800):
            print("{} {} {}".format(t1, t2, t3))
            print(count)
            print_count+= 1
            break
        count += 1

def question_9_linear():
    path = ''
    images, labels = load_mnist(dataset="training", path=path,)

    #shuffle
    start = time.time()
    #p = np.random.permutation(len(labels))
    #a = images[p]
    #b = labels[p]
    #print(a[0])
    #print(b[0])
    total_mse = []
    total_wrong = []
    for n in range(20):
        print("n is {}".format(n))
        p = np.random.permutation(len(labels))
        shuffle_images = images[p]
        shuffle_labels = labels[p]
        #print(a[0])
        #print(shuffle_labels[0])
        #print(shuffle_images.shape)
        #train data
        #total 6500 training data
        #train_data = np.empty((650, 28, 28))
        train_data = []
        train_labels = []
        print("shuffle_labels")
        #print(train_data.shape)
        #print(shuffle_images[0])
        count = [0] * 10
        total_count = 0
        for i in range(len(shuffle_labels)):
            if count[shuffle_labels[i]] < 650:
                #train_data = np.append(train_data, shuffle_images[i])
                train_data.append(shuffle_images[i])
                #print(shuffle_labels[i])
                train_labels.append(shuffle_labels[i])
                count[shuffle_labels[i]] += 1
                total_count += 1
                #print(total_count)
                if total_count == 6500:
                    break
        train_labels = np.array(train_labels)
        train_data= np.array(train_data)
        print("train_labels")
        #print(type(train_labels))
        #print(train_labels)
        #print(train_data.shape)
        nsameples, nx, ny = train_data.shape
        train_image = train_data.reshape((nsameples, nx*ny))
        lin_clf = svm.SVC(kernel='linear', C=10)
        print(lin_clf)
        lin_clf.fit(train_image, train_labels)

        #test data
        #total 650
        print("test")
        test_images, test_labels = load_mnist(dataset="testing", path=path,)
        q = np.random.permutation(len(test_labels))
        shuffle_images = test_images[q]
        shuffle_labels = test_labels[q]
        test_data = []
        test_labels = []
        count = [0] * 10
        total_count = 0
        for i in range(len(shuffle_labels)):
            if count[shuffle_labels[i]] < 163:
                #train_data = np.append(train_data, shuffle_images[i])
                test_data.append(shuffle_images[i])
                #print(shuffle_labels[i])
                test_labels.append(shuffle_labels[i])
                count[shuffle_labels[i]] += 1
                total_count += 1
                #print(total_count)
                if total_count == 1630:
                    break
        test_labels = np.array(test_labels)
        test_data= np.array(test_data)
        nsameples, nx, ny = test_data.shape
        test_data = test_data.reshape((nsameples, nx*ny))
        prediction = lin_clf.predict(test_data)
        print("test and prediction")
        #print(test_labels)
        #print(prediction)
        mse = (np.square(test_labels - prediction)).mean(axis=None)
        print("mse")
        print(mse)
        total_mse.append(mse)
        wrong = 0
        for i in range(1630):
            if test_labels[i] != prediction[i]:
                wrong+=1
        print("wrong")
        wrong /= 1630
        print(wrong)
        total_wrong.append(wrong)

    print("total mse")
    print(total_mse)
    print("mean mse")
    print(statistics.mean(total_mse) )
    print("standard deviation")
    print(statistics.stdev(total_mse))
    print("total wrong")
    print(total_wrong)
    print("mean wrong")
    print(statistics.mean(total_wrong) )
    print("standard deviation")
    print(statistics.stdev(total_wrong))
    end = time.time()
    train_time = end - start
    print("time {}".format(train_time))

def question_9_poly():
    path = ''
    images, labels = load_mnist(dataset="training", path=path,)

    #shuffle
    start = time.time()
    #p = np.random.permutation(len(labels))
    #a = images[p]
    #b = labels[p]
    #print(a[0])
    #print(b[0])
    total_mse = []    
    total_wrong = []
    for n in range(20):
        print("n is {}".format(n))
        p = np.random.permutation(len(labels))
        shuffle_images = images[p]
        shuffle_labels = labels[p]
        #print(a[0])
        #print(shuffle_labels[0])
        #print(shuffle_images.shape)
        #train data
        #total 6500 training data
        #train_data = np.empty((650, 28, 28))
        train_data = []
        train_labels = []
        print("shuffle_labels")
        #print(train_data.shape)
        #print(shuffle_images[0])
        count = [0] * 10
        total_count = 0
        for i in range(len(shuffle_labels)):
            if count[shuffle_labels[i]] < 650:
                #train_data = np.append(train_data, shuffle_images[i])
                train_data.append(shuffle_images[i])
                #print(shuffle_labels[i])
                train_labels.append(shuffle_labels[i])
                count[shuffle_labels[i]] += 1
                total_count += 1
                #print(total_count)
                if total_count == 6500:
                    break
        train_labels = np.array(train_labels)
        train_data= np.array(train_data)
        print("train_labels")
        #print(type(train_labels))
        #print(train_labels)
        #print(train_data.shape)
        nsameples, nx, ny = train_data.shape
        train_image = train_data.reshape((nsameples, nx*ny))
        lin_clf = svm.SVC(kernel='poly', C=0.1)
        print(lin_clf)
        lin_clf.fit(train_image, train_labels)

        #test data
        #total 650
        print("test")
        test_images, test_labels = load_mnist(dataset="testing", path=path,)
        q = np.random.permutation(len(test_labels))
        shuffle_images = test_images[q]
        shuffle_labels = test_labels[q]
        test_data = []
        test_labels = []
        count = [0] * 10
        total_count = 0
        for i in range(len(shuffle_labels)):
            if count[shuffle_labels[i]] < 163:
                #train_data = np.append(train_data, shuffle_images[i])
                test_data.append(shuffle_images[i])
                #print(shuffle_labels[i])
                test_labels.append(shuffle_labels[i])
                count[shuffle_labels[i]] += 1
                total_count += 1
                #print(total_count)
                if total_count == 1630:
                    break
        test_labels = np.array(test_labels)
        test_data= np.array(test_data)
        nsameples, nx, ny = test_data.shape
        test_data = test_data.reshape((nsameples, nx*ny))
        prediction = lin_clf.predict(test_data)
        print("test and prediction")
        #print(test_labels)
        #print(prediction)
        mse = (np.square(test_labels - prediction)).mean(axis=None)
        print("mse")
        print(mse)
        total_mse.append(mse)
        wrong = 0
        for i in range(1630):
            if test_labels[i] != prediction[i]:
                wrong+=1
        print("wrong")
        wrong /= 1630
        print(wrong)
        total_wrong.append(wrong)

    print("total mse")
    print(total_mse)
    print("mean mse")
    print(statistics.mean(total_mse) )
    print("standard deviation")
    print(statistics.stdev(total_mse))
    print("total wrong")
    print(total_wrong)
    print("mean wrong")
    print(statistics.mean(total_wrong) )
    print("standard deviation")
    print(statistics.stdev(total_wrong))
    end = time.time()
    train_time = end - start
    print("time {}".format(train_time))
    
def question_9_rbf():
    path = ''
    images, labels = load_mnist(dataset="training", path=path,)

    #shuffle
    start = time.time()
    #p = np.random.permutation(len(labels))
    #a = images[p]
    #b = labels[p]
    #print(a[0])
    #print(b[0])
    total_mse = []
    total_wrong = []
    for n in range(20):
        print("n is {}".format(n))
        p = np.random.permutation(len(labels))
        shuffle_images = images[p]
        shuffle_labels = labels[p]
        #print(a[0])
        #print(shuffle_labels[0])
        #print(shuffle_images.shape)
        #train data
        #total 6500 training data
        #train_data = np.empty((650, 28, 28))
        train_data = []
        train_labels = []
        print("shuffle_labels")
        #print(train_data.shape)
        #print(shuffle_images[0])
        count = [0] * 10
        total_count = 0
        for i in range(len(shuffle_labels)):
            if count[shuffle_labels[i]] < 650:
                #train_data = np.append(train_data, shuffle_images[i])
                train_data.append(shuffle_images[i])
                #print(shuffle_labels[i])
                train_labels.append(shuffle_labels[i])
                count[shuffle_labels[i]] += 1
                total_count += 1
                #print(total_count)
                if total_count == 6500:
                    break
        train_labels = np.array(train_labels)
        train_data= np.array(train_data)
        print("train_labels")
        #print(type(train_labels))
        #print(train_labels)
        #print(train_data.shape)
        nsameples, nx, ny = train_data.shape
        train_image = train_data.reshape((nsameples, nx*ny))
        lin_clf = svm.SVC(kernel='rbf', C=10)
        print(lin_clf)
        lin_clf.fit(train_image, train_labels)

        #test data
        #total 1630
        print("test")
        test_images, test_labels = load_mnist(dataset="testing", path=path,)
        q = np.random.permutation(len(test_labels))
        shuffle_images = test_images[q]
        shuffle_labels = test_labels[q]
        test_data = []
        test_labels = []
        count = [0] * 10
        total_count = 0
        for i in range(len(shuffle_labels)):
            if count[shuffle_labels[i]] < 163:
                #train_data = np.append(train_data, shuffle_images[i])
                test_data.append(shuffle_images[i])
                #print(shuffle_labels[i])
                test_labels.append(shuffle_labels[i])
                count[shuffle_labels[i]] += 1
                total_count += 1
                #print(total_count)
                if total_count == 1630:
                    break
        test_labels = np.array(test_labels)
        test_data= np.array(test_data)
        nsameples, nx, ny = test_data.shape
        test_data = test_data.reshape((nsameples, nx*ny))
        prediction = lin_clf.predict(test_data)
        print("test and prediction")
        #print(test_labels)
        #print(prediction)
        mse = (np.square(test_labels - prediction)).mean(axis=None)
        print("mse")
        print(mse)
        total_mse.append(mse)
        wrong = 0
        for i in range(1630):
            if test_labels[i] != prediction[i]:
                wrong+=1
        print("wrong")
        wrong /= 1630
        print(wrong)
        total_wrong.append(wrong)

    print("total mse")
    print(total_mse)
    print("mean mse")
    print(statistics.mean(total_mse) )
    print("standard deviation")
    print(statistics.stdev(total_mse))
    print("total wrong")
    print(total_wrong)
    print("mean wrong")
    print(statistics.mean(total_wrong) )
    print("standard deviation")
    print(statistics.stdev(total_wrong))
    end = time.time()
    train_time = end - start
    print("time {}".format(train_time))

def question_10():
    #data
    #MSE
    #linear_0_1 = [1.1251533742331288, 1.0791411042944785, 1.388957055214724, 1.4220858895705522, 1.2190184049079755, 1.449079754601227, 1.0, 1.0386503067484663, 1.5171779141104293, 1.0852760736196319, 1.3687116564417179, 1.2625766871165645, 1.354601226993865, 1.2361963190184049, 1.174233128834356, 1.3680981595092025, 1.2165644171779142, 1.307361963190184, 1.0533742331288343, 1.152760736196319]
    #linear_1 = [1.323926380368098, 1.3042944785276074, 1.31840490797546, 1.2447852760736196, 1.3006134969325154, 1.5803680981595092, 1.478527607361963, 1.4828220858895707, 1.3177914110429447, 1.2595092024539878, 1.396319018404908, 1.3773006134969326, 1.1791411042944786, 1.4049079754601228, 1.7337423312883435, 1.6631901840490797, 1.4987730061349693, 1.6282208588957054, 1.5447852760736196, 1.641717791411043]
    #linear_10 = [1.5, 1.30920245398773, 1.3300613496932516, 1.5607361963190185, 1.3650306748466257, 1.483435582822086, 1.7607361963190185, 1.2539877300613498, 1.3116564417177914, 1.343558282208589, 1.4613496932515337, 1.4877300613496933, 1.4171779141104295, 1.707361963190184, 1.5042944785276073, 1.5257668711656442, 1.4705521472392638, 1.526993865030675, 1.5098159509202453, 1.5674846625766872]
    #poly_0_1 = [18.612883435582823, 17.424539877300614, 17.730674846625767, 18.52638036809816, 18.01042944785276, 18.21288343558282, 18.150920245398773, 18.06073619631902, 18.106134969325154, 17.64723926380368, 17.91472392638037, 18.08527607361963, 17.82760736196319, 17.71472392638037, 18.022699386503067, 17.007361963190185, 17.726993865030675, 18.107361963190183, 18.332515337423313, 18.239877300613497]
    #poly_1 = [17.955828220858894, 18.44355828220859, 17.783435582822086, 18.019018404907975, 17.68036809815951, 17.37914110429448, 17.823312883435584, 17.580981595092023, 18.020858895705523, 18.479754601226993, 17.97116564417178, 17.67914110429448, 17.550920245398775, 17.459509202453987, 17.753374233128834, 18.496932515337424, 17.37239263803681, 18.03926380368098, 17.28527607361963, 18.759509202453987]
    #poly_10 = [7.75398773006135, 7.158895705521473, 7.976073619631902, 7.001840490797546, 7.984049079754601, 8.030061349693252, 7.893865030674847, 8.042331288343558, 8.512269938650308, 7.6214723926380366, 7.6730061349693255, 7.4822085889570555, 7.478527607361963, 8.233742331288344, 7.840490797546012, 7.408588957055215, 7.707975460122699, 7.360736196319018, 8.023312883435583, 7.382208588957055]
    #rbf_0_1 = [2.8834355828220857, 2.954601226993865, 2.7858895705521474, 2.431288343558282, 2.4263803680981595, 2.911042944785276, 2.9914110429447853, 2.9521472392638035, 2.741717791411043, 2.898159509202454, 3.011042944785276, 3.0644171779141103, 2.6423312883435583, 2.776687116564417, 2.650920245398773, 2.8030674846625767, 2.79079754601227, 2.4834355828220858, 2.8496932515337425, 2.9938650306748467]
    #rbf_1 = [1.6883435582822086, 1.376073619631902, 1.3404907975460123, 1.4312883435582822, 1.3049079754601227, 1.4343558282208588, 1.2171779141104295, 1.2441717791411042, 1.5932515337423312, 1.3196319018404907, 1.1766871165644173, 1.4460122699386504, 1.471165644171779, 1.5171779141104293, 1.4828220858895707, 1.3631901840490797, 1.7184049079754602, 1.4128834355828221, 1.7423312883435582, 1.8331288343558283]
    #rbf_10 = [1.2546012269938651, 1.026993865030675, 0.8920245398773006, 0.9834355828220859, 1.2147239263803682, 0.8269938650306748, 1.0877300613496932, 1.1650306748466257, 0.9773006134969325, 1.0834355828220859, 1.11840490797546, 1.1484662576687117, 1.0122699386503067, 1.2257668711656442, 1.1, 1.0693251533742332, 1.2337423312883435, 1.354601226993865, 1.1503067484662577, 1.183435582822086]
    #wrong
    linear_0_1 =[0.0687116564417178, 0.07116564417177915, 0.0736196319018405, 0.07791411042944785, 0.0736196319018405, 0.07116564417177915, 0.07300613496932515, 0.07484662576687116, 0.08282208588957055, 0.0656441717791411, 0.0785276073619632, 0.06319018404907975, 0.07607361963190185, 0.0705521472392638, 0.07730061349693251, 0.06625766871165645, 0.0785276073619632, 0.0705521472392638, 0.07239263803680981, 0.0754601226993865]
    linear_1 = [0.08650306748466258, 0.08895705521472393, 0.08282208588957055, 0.08404907975460123, 0.10122699386503067, 0.09202453987730061, 0.09202453987730061, 0.08650306748466258, 0.09570552147239264, 0.09079754601226994, 0.0852760736196319, 0.09202453987730061, 0.08588957055214724, 0.07975460122699386, 0.09202453987730061, 0.07607361963190185, 0.09693251533742331, 0.08834355828220859, 0.09018404907975461, 0.10429447852760736]
    linear_10 = [0.10736196319018405, 0.08282208588957055, 0.08773006134969324, 0.0852760736196319, 0.09447852760736196, 0.08466257668711656, 0.09018404907975461, 0.08834355828220859, 0.08404907975460123, 0.08957055214723926, 0.09693251533742331, 0.08159509202453988, 0.08220858895705521, 0.08159509202453988, 0.08588957055214724, 0.0803680981595092, 0.09202453987730061, 0.07975460122699386, 0.09631901840490797, 0.09263803680981596]
    poly_0_1 = [0.7895705521472393, 0.7901840490797546, 0.7920245398773006, 0.8024539877300614, 0.7858895705521473, 0.7754601226993865, 0.8122699386503067, 0.7975460122699386, 0.7717791411042945, 0.7834355828220859, 0.7901840490797546, 0.7938650306748466, 0.8153374233128834, 0.8134969325153374, 0.7907975460122699, 0.8030674846625767, 0.7871165644171779, 0.8012269938650307, 0.7834355828220859, 0.8171779141104294]
    poly_1 = [0.7705521472392638, 0.7907975460122699, 0.8116564417177914, 0.7693251533742331, 0.7797546012269939, 0.7877300613496933, 0.7644171779141105, 0.7895705521472393, 0.811042944785276, 0.7987730061349694, 0.7613496932515338, 0.7711656441717791, 0.7865030674846626, 0.7926380368098159, 0.7926380368098159, 0.7895705521472393, 0.8012269938650307, 0.7791411042944786, 0.7840490797546013, 0.7963190184049079]
    poly_10 = [0.3619631901840491, 0.35521472392638037, 0.36441717791411044, 0.3588957055214724, 0.33680981595092024, 0.37975460122699384, 0.37239263803680983, 0.37668711656441717, 0.3576687116564417, 0.37300613496932516, 0.35276073619631904, 0.3638036809815951, 0.36319018404907977, 0.37300613496932516, 0.3521472392638037, 0.3588957055214724, 0.3815950920245399, 0.34171779141104297, 0.3570552147239264, 0.34171779141104297]
    rbf_0_1 = [0.14846625766871166, 0.1588957055214724, 0.1460122699386503, 0.13865030674846626, 0.1325153374233129, 0.1521472392638037, 0.15460122699386503, 0.15276073619631902, 0.16441717791411042, 0.15276073619631902, 0.15460122699386503, 0.14969325153374233, 0.15950920245398773, 0.15276073619631902, 0.1570552147239264, 0.14785276073619633, 0.14355828220858896, 0.14478527607361963, 0.15766871165644172, 0.15644171779141106]
    rbf_1 = [0.0687116564417178, 0.0754601226993865, 0.07791411042944785, 0.10122699386503067, 0.09570552147239264, 0.08588957055214724, 0.08282208588957055, 0.0803680981595092, 0.0852760736196319, 0.07914110429447853, 0.08957055214723926, 0.09754601226993866, 0.0803680981595092, 0.08834355828220859, 0.08159509202453988, 0.08588957055214724, 0.08650306748466258, 0.08650306748466258, 0.08466257668711656, 0.07239263803680981]
    rbf_10 =[0.06748466257668712, 0.06625766871165645, 0.0558282208588957, 0.06012269938650307, 0.06073619631901841, 0.06503067484662577, 0.05276073619631902, 0.06441717791411043, 0.0558282208588957, 0.05460122699386503, 0.06073619631901841, 0.06687116564417178, 0.06257668711656442, 0.06073619631901841, 0.06441717791411043, 0.06073619631901841, 0.06441717791411043, 0.0754601226993865, 0.06932515337423313, 0.05889570552147239]

     
    C_0_1 = linear_0_1 + poly_0_1 + rbf_0_1
    C_1 = linear_1 + poly_1 + rbf_1
    C_10 = linear_10 + poly_10 + rbf_10

    #combine data
    #data_to_plot = [linear_0_1, linear_1, linear_10, poly_0_1, poly_1, poly_10, rbf_0_1, rbf_1, rbf_10]
    data_to_plot = [C_0_1, C_1, C_10]

    # Create a figure instance
    fig = plt.figure(1, figsize=(9, 6))
    # Create an axes instance
    ax = fig.add_subplot(111)
    # Create the boxplot
    bp = ax.boxplot(data_to_plot)
    ## Custom x-axis labels
    ax.set_xticklabels(['C = 0.1', 'C = 1', 'C = 10'])
    #n = 60
    plt.figtext(.8, .8, "n = 60")
    #y axis
    plt.ylabel ('wrong/size')
    # Save the figure
    fig.savefig('q10e_wrong.png', bbox_inches='tight')

def question_12():
    #MSE
    # linear_0_1 = [1.1251533742331288, 1.0791411042944785, 1.388957055214724, 1.4220858895705522, 1.2190184049079755, 1.449079754601227, 1.0, 1.0386503067484663, 1.5171779141104293, 1.0852760736196319, 1.3687116564417179, 1.2625766871165645, 1.354601226993865, 1.2361963190184049, 1.174233128834356, 1.3680981595092025, 1.2165644171779142, 1.307361963190184, 1.0533742331288343, 1.152760736196319]
    # linear_1 = [1.323926380368098, 1.3042944785276074, 1.31840490797546, 1.2447852760736196, 1.3006134969325154, 1.5803680981595092, 1.478527607361963, 1.4828220858895707, 1.3177914110429447, 1.2595092024539878, 1.396319018404908, 1.3773006134969326, 1.1791411042944786, 1.4049079754601228, 1.7337423312883435, 1.6631901840490797, 1.4987730061349693, 1.6282208588957054, 1.5447852760736196, 1.641717791411043]
    # linear_10 = [1.5, 1.30920245398773, 1.3300613496932516, 1.5607361963190185, 1.3650306748466257, 1.483435582822086, 1.7607361963190185, 1.2539877300613498, 1.3116564417177914, 1.343558282208589, 1.4613496932515337, 1.4877300613496933, 1.4171779141104295, 1.707361963190184, 1.5042944785276073, 1.5257668711656442, 1.4705521472392638, 1.526993865030675, 1.5098159509202453, 1.5674846625766872]
    # poly_0_1 = [18.612883435582823, 17.424539877300614, 17.730674846625767, 18.52638036809816, 18.01042944785276, 18.21288343558282, 18.150920245398773, 18.06073619631902, 18.106134969325154, 17.64723926380368, 17.91472392638037, 18.08527607361963, 17.82760736196319, 17.71472392638037, 18.022699386503067, 17.007361963190185, 17.726993865030675, 18.107361963190183, 18.332515337423313, 18.239877300613497]
    # poly_1 = [17.955828220858894, 18.44355828220859, 17.783435582822086, 18.019018404907975, 17.68036809815951, 17.37914110429448, 17.823312883435584, 17.580981595092023, 18.020858895705523, 18.479754601226993, 17.97116564417178, 17.67914110429448, 17.550920245398775, 17.459509202453987, 17.753374233128834, 18.496932515337424, 17.37239263803681, 18.03926380368098, 17.28527607361963, 18.759509202453987]
    # poly_10 = [7.75398773006135, 7.158895705521473, 7.976073619631902, 7.001840490797546, 7.984049079754601, 8.030061349693252, 7.893865030674847, 8.042331288343558, 8.512269938650308, 7.6214723926380366, 7.6730061349693255, 7.4822085889570555, 7.478527607361963, 8.233742331288344, 7.840490797546012, 7.408588957055215, 7.707975460122699, 7.360736196319018, 8.023312883435583, 7.382208588957055]
    # rbf_0_1 = [2.8834355828220857, 2.954601226993865, 2.7858895705521474, 2.431288343558282, 2.4263803680981595, 2.911042944785276, 2.9914110429447853, 2.9521472392638035, 2.741717791411043, 2.898159509202454, 3.011042944785276, 3.0644171779141103, 2.6423312883435583, 2.776687116564417, 2.650920245398773, 2.8030674846625767, 2.79079754601227, 2.4834355828220858, 2.8496932515337425, 2.9938650306748467]
    # rbf_1 = [1.6883435582822086, 1.376073619631902, 1.3404907975460123, 1.4312883435582822, 1.3049079754601227, 1.4343558282208588, 1.2171779141104295, 1.2441717791411042, 1.5932515337423312, 1.3196319018404907, 1.1766871165644173, 1.4460122699386504, 1.471165644171779, 1.5171779141104293, 1.4828220858895707, 1.3631901840490797, 1.7184049079754602, 1.4128834355828221, 1.7423312883435582, 1.8331288343558283]
    # rbf_10 = [1.2546012269938651, 1.026993865030675, 0.8920245398773006, 0.9834355828220859, 1.2147239263803682, 0.8269938650306748, 1.0877300613496932, 1.1650306748466257, 0.9773006134969325, 1.0834355828220859, 1.11840490797546, 1.1484662576687117, 1.0122699386503067, 1.2257668711656442, 1.1, 1.0693251533742332, 1.2337423312883435, 1.354601226993865, 1.1503067484662577, 1.183435582822086]
    #wrong/size
    linear_0_1 =[0.0687116564417178, 0.07116564417177915, 0.0736196319018405, 0.07791411042944785, 0.0736196319018405, 0.07116564417177915, 0.07300613496932515, 0.07484662576687116, 0.08282208588957055, 0.0656441717791411, 0.0785276073619632, 0.06319018404907975, 0.07607361963190185, 0.0705521472392638, 0.07730061349693251, 0.06625766871165645, 0.0785276073619632, 0.0705521472392638, 0.07239263803680981, 0.0754601226993865]
    linear_1 = [0.08650306748466258, 0.08895705521472393, 0.08282208588957055, 0.08404907975460123, 0.10122699386503067, 0.09202453987730061, 0.09202453987730061, 0.08650306748466258, 0.09570552147239264, 0.09079754601226994, 0.0852760736196319, 0.09202453987730061, 0.08588957055214724, 0.07975460122699386, 0.09202453987730061, 0.07607361963190185, 0.09693251533742331, 0.08834355828220859, 0.09018404907975461, 0.10429447852760736]
    linear_10 = [0.10736196319018405, 0.08282208588957055, 0.08773006134969324, 0.0852760736196319, 0.09447852760736196, 0.08466257668711656, 0.09018404907975461, 0.08834355828220859, 0.08404907975460123, 0.08957055214723926, 0.09693251533742331, 0.08159509202453988, 0.08220858895705521, 0.08159509202453988, 0.08588957055214724, 0.0803680981595092, 0.09202453987730061, 0.07975460122699386, 0.09631901840490797, 0.09263803680981596]
    poly_0_1 = [0.7895705521472393, 0.7901840490797546, 0.7920245398773006, 0.8024539877300614, 0.7858895705521473, 0.7754601226993865, 0.8122699386503067, 0.7975460122699386, 0.7717791411042945, 0.7834355828220859, 0.7901840490797546, 0.7938650306748466, 0.8153374233128834, 0.8134969325153374, 0.7907975460122699, 0.8030674846625767, 0.7871165644171779, 0.8012269938650307, 0.7834355828220859, 0.8171779141104294]
    poly_1 = [0.7705521472392638, 0.7907975460122699, 0.8116564417177914, 0.7693251533742331, 0.7797546012269939, 0.7877300613496933, 0.7644171779141105, 0.7895705521472393, 0.811042944785276, 0.7987730061349694, 0.7613496932515338, 0.7711656441717791, 0.7865030674846626, 0.7926380368098159, 0.7926380368098159, 0.7895705521472393, 0.8012269938650307, 0.7791411042944786, 0.7840490797546013, 0.7963190184049079]
    poly_10 = [0.3619631901840491, 0.35521472392638037, 0.36441717791411044, 0.3588957055214724, 0.33680981595092024, 0.37975460122699384, 0.37239263803680983, 0.37668711656441717, 0.3576687116564417, 0.37300613496932516, 0.35276073619631904, 0.3638036809815951, 0.36319018404907977, 0.37300613496932516, 0.3521472392638037, 0.3588957055214724, 0.3815950920245399, 0.34171779141104297, 0.3570552147239264, 0.34171779141104297]
    rbf_0_1 = [0.14846625766871166, 0.1588957055214724, 0.1460122699386503, 0.13865030674846626, 0.1325153374233129, 0.1521472392638037, 0.15460122699386503, 0.15276073619631902, 0.16441717791411042, 0.15276073619631902, 0.15460122699386503, 0.14969325153374233, 0.15950920245398773, 0.15276073619631902, 0.1570552147239264, 0.14785276073619633, 0.14355828220858896, 0.14478527607361963, 0.15766871165644172, 0.15644171779141106]
    rbf_1 = [0.0687116564417178, 0.0754601226993865, 0.07791411042944785, 0.10122699386503067, 0.09570552147239264, 0.08588957055214724, 0.08282208588957055, 0.0803680981595092, 0.0852760736196319, 0.07914110429447853, 0.08957055214723926, 0.09754601226993866, 0.0803680981595092, 0.08834355828220859, 0.08159509202453988, 0.08588957055214724, 0.08650306748466258, 0.08650306748466258, 0.08466257668711656, 0.07239263803680981]
    rbf_10 =[0.06748466257668712, 0.06625766871165645, 0.0558282208588957, 0.06012269938650307, 0.06073619631901841, 0.06503067484662577, 0.05276073619631902, 0.06441717791411043, 0.0558282208588957, 0.05460122699386503, 0.06073619631901841, 0.06687116564417178, 0.06257668711656442, 0.06073619631901841, 0.06441717791411043, 0.06073619631901841, 0.06441717791411043, 0.0754601226993865, 0.06932515337423313, 0.05889570552147239]

    C_0_1 = linear_0_1 + poly_0_1 + rbf_0_1
    C_1 = linear_1 + poly_1 + rbf_1
    C_10 = linear_10 + poly_10 + rbf_10

    print("0.1 vs 1")
    print(mannwhitneyu(C_0_1, C_1))
    print("1 vs 10")
    print(mannwhitneyu(C_1, C_10))
    print("0.1 vs 10")
    print(mannwhitneyu(C_0_1, C_10))
    

def question_13():
    #data
    #MSE
    # linear_0_1 = [1.1251533742331288, 1.0791411042944785, 1.388957055214724, 1.4220858895705522, 1.2190184049079755, 1.449079754601227, 1.0, 1.0386503067484663, 1.5171779141104293, 1.0852760736196319, 1.3687116564417179, 1.2625766871165645, 1.354601226993865, 1.2361963190184049, 1.174233128834356, 1.3680981595092025, 1.2165644171779142, 1.307361963190184, 1.0533742331288343, 1.152760736196319]
    # linear_1 = [1.323926380368098, 1.3042944785276074, 1.31840490797546, 1.2447852760736196, 1.3006134969325154, 1.5803680981595092, 1.478527607361963, 1.4828220858895707, 1.3177914110429447, 1.2595092024539878, 1.396319018404908, 1.3773006134969326, 1.1791411042944786, 1.4049079754601228, 1.7337423312883435, 1.6631901840490797, 1.4987730061349693, 1.6282208588957054, 1.5447852760736196, 1.641717791411043]
    # linear_10 = [1.5, 1.30920245398773, 1.3300613496932516, 1.5607361963190185, 1.3650306748466257, 1.483435582822086, 1.7607361963190185, 1.2539877300613498, 1.3116564417177914, 1.343558282208589, 1.4613496932515337, 1.4877300613496933, 1.4171779141104295, 1.707361963190184, 1.5042944785276073, 1.5257668711656442, 1.4705521472392638, 1.526993865030675, 1.5098159509202453, 1.5674846625766872]
    # poly_0_1 = [18.612883435582823, 17.424539877300614, 17.730674846625767, 18.52638036809816, 18.01042944785276, 18.21288343558282, 18.150920245398773, 18.06073619631902, 18.106134969325154, 17.64723926380368, 17.91472392638037, 18.08527607361963, 17.82760736196319, 17.71472392638037, 18.022699386503067, 17.007361963190185, 17.726993865030675, 18.107361963190183, 18.332515337423313, 18.239877300613497]
    # poly_1 = [17.955828220858894, 18.44355828220859, 17.783435582822086, 18.019018404907975, 17.68036809815951, 17.37914110429448, 17.823312883435584, 17.580981595092023, 18.020858895705523, 18.479754601226993, 17.97116564417178, 17.67914110429448, 17.550920245398775, 17.459509202453987, 17.753374233128834, 18.496932515337424, 17.37239263803681, 18.03926380368098, 17.28527607361963, 18.759509202453987]
    # poly_10 = [7.75398773006135, 7.158895705521473, 7.976073619631902, 7.001840490797546, 7.984049079754601, 8.030061349693252, 7.893865030674847, 8.042331288343558, 8.512269938650308, 7.6214723926380366, 7.6730061349693255, 7.4822085889570555, 7.478527607361963, 8.233742331288344, 7.840490797546012, 7.408588957055215, 7.707975460122699, 7.360736196319018, 8.023312883435583, 7.382208588957055]
    # rbf_0_1 = [2.8834355828220857, 2.954601226993865, 2.7858895705521474, 2.431288343558282, 2.4263803680981595, 2.911042944785276, 2.9914110429447853, 2.9521472392638035, 2.741717791411043, 2.898159509202454, 3.011042944785276, 3.0644171779141103, 2.6423312883435583, 2.776687116564417, 2.650920245398773, 2.8030674846625767, 2.79079754601227, 2.4834355828220858, 2.8496932515337425, 2.9938650306748467]
    # rbf_1 = [1.6883435582822086, 1.376073619631902, 1.3404907975460123, 1.4312883435582822, 1.3049079754601227, 1.4343558282208588, 1.2171779141104295, 1.2441717791411042, 1.5932515337423312, 1.3196319018404907, 1.1766871165644173, 1.4460122699386504, 1.471165644171779, 1.5171779141104293, 1.4828220858895707, 1.3631901840490797, 1.7184049079754602, 1.4128834355828221, 1.7423312883435582, 1.8331288343558283]
    # rbf_10 = [1.2546012269938651, 1.026993865030675, 0.8920245398773006, 0.9834355828220859, 1.2147239263803682, 0.8269938650306748, 1.0877300613496932, 1.1650306748466257, 0.9773006134969325, 1.0834355828220859, 1.11840490797546, 1.1484662576687117, 1.0122699386503067, 1.2257668711656442, 1.1, 1.0693251533742332, 1.2337423312883435, 1.354601226993865, 1.1503067484662577, 1.183435582822086]
    #wrong/size
    linear_0_1 =[0.0687116564417178, 0.07116564417177915, 0.0736196319018405, 0.07791411042944785, 0.0736196319018405, 0.07116564417177915, 0.07300613496932515, 0.07484662576687116, 0.08282208588957055, 0.0656441717791411, 0.0785276073619632, 0.06319018404907975, 0.07607361963190185, 0.0705521472392638, 0.07730061349693251, 0.06625766871165645, 0.0785276073619632, 0.0705521472392638, 0.07239263803680981, 0.0754601226993865]
    linear_1 = [0.08650306748466258, 0.08895705521472393, 0.08282208588957055, 0.08404907975460123, 0.10122699386503067, 0.09202453987730061, 0.09202453987730061, 0.08650306748466258, 0.09570552147239264, 0.09079754601226994, 0.0852760736196319, 0.09202453987730061, 0.08588957055214724, 0.07975460122699386, 0.09202453987730061, 0.07607361963190185, 0.09693251533742331, 0.08834355828220859, 0.09018404907975461, 0.10429447852760736]
    linear_10 = [0.10736196319018405, 0.08282208588957055, 0.08773006134969324, 0.0852760736196319, 0.09447852760736196, 0.08466257668711656, 0.09018404907975461, 0.08834355828220859, 0.08404907975460123, 0.08957055214723926, 0.09693251533742331, 0.08159509202453988, 0.08220858895705521, 0.08159509202453988, 0.08588957055214724, 0.0803680981595092, 0.09202453987730061, 0.07975460122699386, 0.09631901840490797, 0.09263803680981596]
    poly_0_1 = [0.7895705521472393, 0.7901840490797546, 0.7920245398773006, 0.8024539877300614, 0.7858895705521473, 0.7754601226993865, 0.8122699386503067, 0.7975460122699386, 0.7717791411042945, 0.7834355828220859, 0.7901840490797546, 0.7938650306748466, 0.8153374233128834, 0.8134969325153374, 0.7907975460122699, 0.8030674846625767, 0.7871165644171779, 0.8012269938650307, 0.7834355828220859, 0.8171779141104294]
    poly_1 = [0.7705521472392638, 0.7907975460122699, 0.8116564417177914, 0.7693251533742331, 0.7797546012269939, 0.7877300613496933, 0.7644171779141105, 0.7895705521472393, 0.811042944785276, 0.7987730061349694, 0.7613496932515338, 0.7711656441717791, 0.7865030674846626, 0.7926380368098159, 0.7926380368098159, 0.7895705521472393, 0.8012269938650307, 0.7791411042944786, 0.7840490797546013, 0.7963190184049079]
    poly_10 = [0.3619631901840491, 0.35521472392638037, 0.36441717791411044, 0.3588957055214724, 0.33680981595092024, 0.37975460122699384, 0.37239263803680983, 0.37668711656441717, 0.3576687116564417, 0.37300613496932516, 0.35276073619631904, 0.3638036809815951, 0.36319018404907977, 0.37300613496932516, 0.3521472392638037, 0.3588957055214724, 0.3815950920245399, 0.34171779141104297, 0.3570552147239264, 0.34171779141104297]
    rbf_0_1 = [0.14846625766871166, 0.1588957055214724, 0.1460122699386503, 0.13865030674846626, 0.1325153374233129, 0.1521472392638037, 0.15460122699386503, 0.15276073619631902, 0.16441717791411042, 0.15276073619631902, 0.15460122699386503, 0.14969325153374233, 0.15950920245398773, 0.15276073619631902, 0.1570552147239264, 0.14785276073619633, 0.14355828220858896, 0.14478527607361963, 0.15766871165644172, 0.15644171779141106]
    rbf_1 = [0.0687116564417178, 0.0754601226993865, 0.07791411042944785, 0.10122699386503067, 0.09570552147239264, 0.08588957055214724, 0.08282208588957055, 0.0803680981595092, 0.0852760736196319, 0.07914110429447853, 0.08957055214723926, 0.09754601226993866, 0.0803680981595092, 0.08834355828220859, 0.08159509202453988, 0.08588957055214724, 0.08650306748466258, 0.08650306748466258, 0.08466257668711656, 0.07239263803680981]
    rbf_10 =[0.06748466257668712, 0.06625766871165645, 0.0558282208588957, 0.06012269938650307, 0.06073619631901841, 0.06503067484662577, 0.05276073619631902, 0.06441717791411043, 0.0558282208588957, 0.05460122699386503, 0.06073619631901841, 0.06687116564417178, 0.06257668711656442, 0.06073619631901841, 0.06441717791411043, 0.06073619631901841, 0.06441717791411043, 0.0754601226993865, 0.06932515337423313, 0.05889570552147239]

    linear = linear_0_1 + linear_1 + linear_10
    poly = poly_0_1 + poly_1 + poly_10
    rbf = rbf_0_1 + rbf_1 + rbf_10

    #combine data
    #data_to_plot = [linear_0_1, linear_1, linear_10, poly_0_1, poly_1, poly_10, rbf_0_1, rbf_1, rbf_10]
    data_to_plot = [linear, poly, rbf]

    # Create a figure instance
    fig = plt.figure(1, figsize=(9, 6))
    # Create an axes instance
    ax = fig.add_subplot(111)
    # Create the boxplot
    bp = ax.boxplot(data_to_plot)
    ## Custom x-axis labels
    ax.set_xticklabels(['linear', 'poly', 'rbf'])
    #n = 60
    plt.figtext(.8, .8, "n = 60")
    #y axis
    plt.ylabel ('wrong/size')
    # Save the figure
    fig.savefig('q13_e_wrong.png', bbox_inches='tight')

def question_15():
    #MSE
    # linear_0_1 = [1.1251533742331288, 1.0791411042944785, 1.388957055214724, 1.4220858895705522, 1.2190184049079755, 1.449079754601227, 1.0, 1.0386503067484663, 1.5171779141104293, 1.0852760736196319, 1.3687116564417179, 1.2625766871165645, 1.354601226993865, 1.2361963190184049, 1.174233128834356, 1.3680981595092025, 1.2165644171779142, 1.307361963190184, 1.0533742331288343, 1.152760736196319]
    # linear_1 = [1.323926380368098, 1.3042944785276074, 1.31840490797546, 1.2447852760736196, 1.3006134969325154, 1.5803680981595092, 1.478527607361963, 1.4828220858895707, 1.3177914110429447, 1.2595092024539878, 1.396319018404908, 1.3773006134969326, 1.1791411042944786, 1.4049079754601228, 1.7337423312883435, 1.6631901840490797, 1.4987730061349693, 1.6282208588957054, 1.5447852760736196, 1.641717791411043]
    # linear_10 = [1.5, 1.30920245398773, 1.3300613496932516, 1.5607361963190185, 1.3650306748466257, 1.483435582822086, 1.7607361963190185, 1.2539877300613498, 1.3116564417177914, 1.343558282208589, 1.4613496932515337, 1.4877300613496933, 1.4171779141104295, 1.707361963190184, 1.5042944785276073, 1.5257668711656442, 1.4705521472392638, 1.526993865030675, 1.5098159509202453, 1.5674846625766872]
    # poly_0_1 = [18.612883435582823, 17.424539877300614, 17.730674846625767, 18.52638036809816, 18.01042944785276, 18.21288343558282, 18.150920245398773, 18.06073619631902, 18.106134969325154, 17.64723926380368, 17.91472392638037, 18.08527607361963, 17.82760736196319, 17.71472392638037, 18.022699386503067, 17.007361963190185, 17.726993865030675, 18.107361963190183, 18.332515337423313, 18.239877300613497]
    # poly_1 = [17.955828220858894, 18.44355828220859, 17.783435582822086, 18.019018404907975, 17.68036809815951, 17.37914110429448, 17.823312883435584, 17.580981595092023, 18.020858895705523, 18.479754601226993, 17.97116564417178, 17.67914110429448, 17.550920245398775, 17.459509202453987, 17.753374233128834, 18.496932515337424, 17.37239263803681, 18.03926380368098, 17.28527607361963, 18.759509202453987]
    # poly_10 = [7.75398773006135, 7.158895705521473, 7.976073619631902, 7.001840490797546, 7.984049079754601, 8.030061349693252, 7.893865030674847, 8.042331288343558, 8.512269938650308, 7.6214723926380366, 7.6730061349693255, 7.4822085889570555, 7.478527607361963, 8.233742331288344, 7.840490797546012, 7.408588957055215, 7.707975460122699, 7.360736196319018, 8.023312883435583, 7.382208588957055]
    # rbf_0_1 = [2.8834355828220857, 2.954601226993865, 2.7858895705521474, 2.431288343558282, 2.4263803680981595, 2.911042944785276, 2.9914110429447853, 2.9521472392638035, 2.741717791411043, 2.898159509202454, 3.011042944785276, 3.0644171779141103, 2.6423312883435583, 2.776687116564417, 2.650920245398773, 2.8030674846625767, 2.79079754601227, 2.4834355828220858, 2.8496932515337425, 2.9938650306748467]
    # rbf_1 = [1.6883435582822086, 1.376073619631902, 1.3404907975460123, 1.4312883435582822, 1.3049079754601227, 1.4343558282208588, 1.2171779141104295, 1.2441717791411042, 1.5932515337423312, 1.3196319018404907, 1.1766871165644173, 1.4460122699386504, 1.471165644171779, 1.5171779141104293, 1.4828220858895707, 1.3631901840490797, 1.7184049079754602, 1.4128834355828221, 1.7423312883435582, 1.8331288343558283]
    # rbf_10 = [1.2546012269938651, 1.026993865030675, 0.8920245398773006, 0.9834355828220859, 1.2147239263803682, 0.8269938650306748, 1.0877300613496932, 1.1650306748466257, 0.9773006134969325, 1.0834355828220859, 1.11840490797546, 1.1484662576687117, 1.0122699386503067, 1.2257668711656442, 1.1, 1.0693251533742332, 1.2337423312883435, 1.354601226993865, 1.1503067484662577, 1.183435582822086]
    #wrong/size
    linear_0_1 =[0.0687116564417178, 0.07116564417177915, 0.0736196319018405, 0.07791411042944785, 0.0736196319018405, 0.07116564417177915, 0.07300613496932515, 0.07484662576687116, 0.08282208588957055, 0.0656441717791411, 0.0785276073619632, 0.06319018404907975, 0.07607361963190185, 0.0705521472392638, 0.07730061349693251, 0.06625766871165645, 0.0785276073619632, 0.0705521472392638, 0.07239263803680981, 0.0754601226993865]
    linear_1 = [0.08650306748466258, 0.08895705521472393, 0.08282208588957055, 0.08404907975460123, 0.10122699386503067, 0.09202453987730061, 0.09202453987730061, 0.08650306748466258, 0.09570552147239264, 0.09079754601226994, 0.0852760736196319, 0.09202453987730061, 0.08588957055214724, 0.07975460122699386, 0.09202453987730061, 0.07607361963190185, 0.09693251533742331, 0.08834355828220859, 0.09018404907975461, 0.10429447852760736]
    linear_10 = [0.10736196319018405, 0.08282208588957055, 0.08773006134969324, 0.0852760736196319, 0.09447852760736196, 0.08466257668711656, 0.09018404907975461, 0.08834355828220859, 0.08404907975460123, 0.08957055214723926, 0.09693251533742331, 0.08159509202453988, 0.08220858895705521, 0.08159509202453988, 0.08588957055214724, 0.0803680981595092, 0.09202453987730061, 0.07975460122699386, 0.09631901840490797, 0.09263803680981596]
    poly_0_1 = [0.7895705521472393, 0.7901840490797546, 0.7920245398773006, 0.8024539877300614, 0.7858895705521473, 0.7754601226993865, 0.8122699386503067, 0.7975460122699386, 0.7717791411042945, 0.7834355828220859, 0.7901840490797546, 0.7938650306748466, 0.8153374233128834, 0.8134969325153374, 0.7907975460122699, 0.8030674846625767, 0.7871165644171779, 0.8012269938650307, 0.7834355828220859, 0.8171779141104294]
    poly_1 = [0.7705521472392638, 0.7907975460122699, 0.8116564417177914, 0.7693251533742331, 0.7797546012269939, 0.7877300613496933, 0.7644171779141105, 0.7895705521472393, 0.811042944785276, 0.7987730061349694, 0.7613496932515338, 0.7711656441717791, 0.7865030674846626, 0.7926380368098159, 0.7926380368098159, 0.7895705521472393, 0.8012269938650307, 0.7791411042944786, 0.7840490797546013, 0.7963190184049079]
    poly_10 = [0.3619631901840491, 0.35521472392638037, 0.36441717791411044, 0.3588957055214724, 0.33680981595092024, 0.37975460122699384, 0.37239263803680983, 0.37668711656441717, 0.3576687116564417, 0.37300613496932516, 0.35276073619631904, 0.3638036809815951, 0.36319018404907977, 0.37300613496932516, 0.3521472392638037, 0.3588957055214724, 0.3815950920245399, 0.34171779141104297, 0.3570552147239264, 0.34171779141104297]
    rbf_0_1 = [0.14846625766871166, 0.1588957055214724, 0.1460122699386503, 0.13865030674846626, 0.1325153374233129, 0.1521472392638037, 0.15460122699386503, 0.15276073619631902, 0.16441717791411042, 0.15276073619631902, 0.15460122699386503, 0.14969325153374233, 0.15950920245398773, 0.15276073619631902, 0.1570552147239264, 0.14785276073619633, 0.14355828220858896, 0.14478527607361963, 0.15766871165644172, 0.15644171779141106]
    rbf_1 = [0.0687116564417178, 0.0754601226993865, 0.07791411042944785, 0.10122699386503067, 0.09570552147239264, 0.08588957055214724, 0.08282208588957055, 0.0803680981595092, 0.0852760736196319, 0.07914110429447853, 0.08957055214723926, 0.09754601226993866, 0.0803680981595092, 0.08834355828220859, 0.08159509202453988, 0.08588957055214724, 0.08650306748466258, 0.08650306748466258, 0.08466257668711656, 0.07239263803680981]
    rbf_10 =[0.06748466257668712, 0.06625766871165645, 0.0558282208588957, 0.06012269938650307, 0.06073619631901841, 0.06503067484662577, 0.05276073619631902, 0.06441717791411043, 0.0558282208588957, 0.05460122699386503, 0.06073619631901841, 0.06687116564417178, 0.06257668711656442, 0.06073619631901841, 0.06441717791411043, 0.06073619631901841, 0.06441717791411043, 0.0754601226993865, 0.06932515337423313, 0.05889570552147239]

    linear = linear_0_1 + linear_1 + linear_10
    poly = poly_0_1 + poly_1 + poly_10
    rbf = rbf_0_1 + rbf_1 + rbf_10
    
    print("linear vs poly")
    print(mannwhitneyu(linear, poly))
    print("linear vs rbf")
    print(mannwhitneyu(linear, rbf))
    print("poly vs rbf")
    print(mannwhitneyu(poly, rbf))

if __name__ == "__main__":
    #read_training_and_testing()
    #read_0_in_training_and_testing()
    #example_5()    
    #question_6()
    #question_9_linear()
    #question_9_poly()
    #question_9_rbf()
    #question_10()
    # question_12()
    #question_13()
    question_15()