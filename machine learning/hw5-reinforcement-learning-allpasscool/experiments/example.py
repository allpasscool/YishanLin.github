import gym
# from code import MultiArmedBandit
from multi_armed_bandit import *
from q_learning import QLearning
import matplotlib.pyplot as plt
import time
import numpy as np
from slot_machines import *

def q2a():
    first_line = 0
    second_line = 0
    third_line = 0
    for i in range(10):
        env = gym.make('SlotMachines-v0')
        agent = MultiArmedBandit()
        action_values, rewards = agent.fit(env, 100000)
        if i == 0:
            first_line = rewards
            second_line = rewards
            third_line = rewards
        elif i > 0 and i < 5:
            np.add(second_line, rewards)
            np.add(third_line, rewards)
        else:
            np.add(third_line, rewards)
    
    second_line = np.divide(second_line, 5)
    third_line = np.divide(third_line, 10)

    print('Finished example experiment')
    

    label = []
    for i in range(100):
        label.append(i)

    fig = plt.figure()
    ax = plt.axes()
    plt.xlabel("np.floor(steps / 100)")
    plt.ylabel("reward")
    plt.plot(label, first_line, label='First line')
    plt.plot(label, second_line, label='second line')#marker = 'o')
    plt.plot(label, third_line, label="third line")
    plt.legend()
    plt.show()
    fig.savefig('2a.png', bbox_inches='tight')

    """
    # Create a figure instance
    fig = plt.figure(1, figsize=(9, 6))
    # Create an axes instance
    ax = fig.add_subplot(111)
    # Create the boxplot
    bp = ax.boxplot(data_to_plot)
    ## Custom x-axis labels
    #ax.set_xticklabels(['linear', 'poly', 'rbf'])
    #n = 60
    #plt.figtext(.8, .8, "n = 60")
    #y axis
    #plt.ylabel ('wrong/size')
    # Save the figure
    fig.savefig('2a.png', bbox_inches='tight')
    """
def q2b():
    print()

if __name__ == "__main__":
    print('Starting example experiment')
    start = time.time()
    
    q2a()





    end = time.time()
    train_time = end - start
    print("train_time")
    print(train_time)




# if __name__ == "__main__":
#     print('Starting example experiment')

#     env = gym.make('FrozenLake-v0')
#     agent = MultiArmedBandit()
#     action_values, rewards = agent.fit(env)

#     print('Finished example experiment')
