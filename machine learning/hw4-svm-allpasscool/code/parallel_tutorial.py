"""
This is a tutorial on parallelization in python. You do not have to use this
for your assignment, but it may come in handy. There are some advanced python
tricks here, and every effort has been made to explain what is going on.
As always, if you have questions come to office hours, or post on Piazza.

This file is organized into 4 examples of parallelization. The first example
does not parallelize at all, the following three show ways to parallelize a
slow function in three different ways. There are many ways to parallelize code
in python; here we only explore one variant.

There is also two "long" running functions at the bottom. These are the functions
we are trying to parallelize.

You will notice that the main function does nothing. You will have to modify it
to try the examples. Poke around. Try to break it. Figure out how these things
work and convince yourself that they do.

A final note on parallelization: if you have 100 things that need to be run very
quickly, it may be tempting to try to spawn 100 different threads but you will
soon run into the resource limits of your computer. There is a sweet spot between
a high amount of parallelization and your computers resources. You'll have to
experiment to find the right amount.

"""
import time
import random

# python doesn't do multithreading*, but we can use this built-in multiprocessing
# library to simulate multithreading. This is much easier to use!
from multiprocessing.dummy import Pool as ThreadPool


def main():
    # experiment with the different example functions here.

    pass


def example1(n_trials):
    """
    Example 1: Running a function with one arg in a for loop, or serially...

    This should seem familiar to you. This is the case that has no parallelization.
    We run every call to our function serially in a for loop.

    Notice how the printed output displays 1, then 2, then 3, then...

    We will transform this case into a parallelized version of our code as this
    tutorial progresses.

    :param n_trials: number of times our long_function_simple() gets called
    :return:
        "results" a list of random numbers.
    """
    results = []
    for trial in range(n_trials):
        results += long_function_simple(trial)  # += is the same as results.append()

    return results


def example2(n_trials):
    """
    Example 2: Running our function with one arg in parallel with ThreadPool.

    Now let's speed up our same long-running function from example1()
    using parallelization. To parallelize, we will use ThreadPool as imported
    above. What ThreadPool will do is spawn "threads" that each run
    at the same time in a separate process (not thread--the reason why is
    beyond the scope of this tutorial).

    This is the same function call as `example1()`, but now it happens much
    quicker!

    Notice how the printed output displays are not in numerical order
    as they were in the first example. This means that if you have a function
    that is dependent on the results from a previous run, you can *not*
    parallelize that function. Each call to our function is completely
    independent and can be called by itself.

    :param n_trials: number of times our long_function_simple() gets called
    :return:
        "results" a list random numbers.
    """
    n_threads = 4  # we decide the number of threads we want
    pool = ThreadPool(n_threads)  # give that number to our ThreadPool constructor
    results = []  # we need somewhere to store the results

    # this is called a `list comprehension`, it can replace for loops
    labels = [t for t in range(n_trials)]

    # Here is where the parallelization happens!
    # The ThreadPool `map()` function will call `long_function_simple()`
    # `n_threads` number of times and for each one it will give it one
    # value from our `labels` list as the input to that function.
    # Note: we give the name of the function without the parens ()!
    results += pool.map(long_function_simple, labels)

    # Our results are a list of lists, so we can flatten it with another list comprehension:
    # Note: This is a *double* list comprehension. Make sure to notice the order of the
    # the list and sublist, it's a bit tricky...
    results = [item for sublist in results for item in sublist]

    return results


def example3(n_trials):
    """
    Example 3: Running our function with changing and
        unchanging args in parallel with ThreadPool

    In this example, only one arg changes, but our function signature takes
    more than one argument. To do this we use a nested helper function that
    keeps the variables that we don't want to change in scope.

    Other than this little trick, the mechanics of running our function
    in parallel is the same as before.

    :param n_trials: number of times our long_function_with_many_args() gets called
    :return:
        "results" a list random numbers.
    """

    n_threads = 4  # we decide the number of threads we want
    pool = ThreadPool(n_threads)  # give it to our ThreadPool object
    results = []  # we need somewhere to store the results

    # As before, labels will change with each run.
    labels = [t for t in range(n_trials)]

    # But now, we'll keep these three variables static/unchanging
    min_static, max_static = 1, 10
    sleep_time = 10.0

    # This is a nested function. They're not very pretty to use,
    # but because we have a lot of variables that are unchanging
    # this is an easy option to
    def long_function_helper(label):
        # min_static, max_static, sleep_time are all in scope here
        # meaning that we have access to them inside this function.
        # So that means min_static=1, max_static=10, and sleep_time=10.0
        return long_function_with_many_args(label, min_static, max_static, sleep_time)

    # Here is where the parallelization happens!
    # We'll pass `map()` our helper function instead of our real function
    results += pool.map(long_function_helper, labels)

    # Again, our results are a list of lists, so we can flatten it with another list comprehension:
    results = [item for sublist in results for item in sublist]

    return results


def example4(n_trials):
    """
    Example 4: Running our function with many changing args in parallel with ThreadPool

    In this example, we use python's built in `zip()` function to get
    a unique set of arguments for each time that we run our function.

    :param n_trials: number of times our long_function_with_many_args() gets called
    :return:
        "results" a list random numbers.
    """

    n_threads = 4  # we decide the number of threads we want
    pool = ThreadPool(n_threads)  # give it to our ThreadPool object
    results = []  # we need somewhere to store the results

    # lets set up different inputs for each run of our function
    labels = [t for t in range(n_trials)]  # the label changes every time we call it (as before)

    # Lets also vary the min and max of our random numbers, and the amount of time it sleeps
    min_variable = [m for m in range(0, n_trials)]
    max_variable = [m for m in range(n_trials, 2*n_trials)]
    sleep_times = [s*0.1 for s in range(n_trials)]

    # `zip` turns our 4 individual lists into one list of tuples
    # So lists `alph = ['a', 'b', 'c']`,` num = [1, 2, 3]`
    # become this: `[('a', 1), ('b', 2), ('c', 3)]`
    arg_list = [z for z in zip(labels, min_variable, max_variable, sleep_times)]

    # Now set up our pool as before, but instead of `map()`, we use `starmap()`.
    # This function will apply multiple arguments.
    results += pool.starmap(long_function_with_many_args, arg_list)

    # Again, our results are a list of lists, so we can flatten it with another list comprehension:
    results = [item for sublist in results for item in sublist]

    return results


# --------------------------------------------------------------------
#           vvvvv  Here be long running functions!  vvvvv
# --------------------------------------------------------------------


def long_function_simple(label):
    """
    This function takes a long time to run!

    It only takes one argument (`label`) that could be different every time we call it.
    This function prints a statement, sleeps for 10 seconds, and then returns a list containing
    two random numbers between [1, 10].

    :param label: A name for this function. This is used when we print out a status message.
    :return:
        (list): a list with two random integers in the interval [1, 10]
    """
    print('Starting our long but simple function, label {}!'.format(label))
    time.sleep(10.0)
    print('Finished our long but simple function, label {}!'.format(label))
    return [random.randint(1, 10), random.randint(1, 10)]


def long_function_with_many_args(label, min_, max_, sleep_time_sec):
    """
    This function takes a long time to run! And it has multiple arguments!!!

    This function takes many arguments, sleeps and then returns two random
    numbers between [min_, max_]. Wowee!

    :param label: A name for this function. This is used when we print out a status message.
    :param min_: (int) The minimum random number
    :param max_: (int) The maximum random number.
    :param sleep_time_sec: (float) The time that this function will sleep for

    :return:
        (list): a list with two random integers in the interval [min_, max_]
    """
    print('Starting our long process with many args, label {}!'.format(label))
    print('{}: min={}, max={}, sleep_time={}'.format(label, min_, max_, sleep_time_sec))
    time.sleep(sleep_time_sec)
    print('Finished our long process with many args, label {}!'.format(label))
    return [random.randint(min_, max_), random.randint(min_, max_)]


if __name__ == '__main__':
    # This is the `main` call in python
    main()