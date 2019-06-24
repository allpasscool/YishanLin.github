import gym
from code import MultiArmedBandit
#from multi_armed_bandit import MultiArmedBandit
#from q_learning import QLearning
from code import QLearning
import matplotlib.pyplot as plt
import time
import numpy as np
#from slot_machines import *

def q2a():
    first_line = 0
    second_line = 0
    third_line = 0
    for i in range(10):
        print(np.random.rand())
        env = gym.make('SlotMachines-v0')
        env.reset()
        agent = MultiArmedBandit()
        action_values, rewards = agent.fit(env, 100000)
        if i == 0:
            first_line = rewards
            second_line = rewards
            third_line = rewards
        elif i > 0 and i < 5:
            second_line = np.add(second_line, rewards)
            third_line = np.add(third_line, rewards)
        else:
            third_line = np.add(third_line, rewards)
    
    second_line = np.divide(second_line, 5)
    third_line = np.divide(third_line, 10)

    print('Finished example experiment')
    print("first line")
    print(first_line.tolist())
    print("second line")
    print(second_line)
    print("third line")
    print(third_line.tolist())

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



def q2b():
    first_line = 0
    second_line = 0
    third_line = 0
    for i in range(10):
        env = gym.make('SlotMachines-v0')
        agent = QLearning()
        action_values, rewards = agent.fit(env, 100000)
        if i == 0:
            first_line = rewards
            second_line = rewards
            third_line = rewards
        elif i > 0 and i < 5:
            second_line =  np.add(second_line, rewards)
            third_line = np.add(third_line, rewards)
        else:
            third_line = np.add(third_line, rewards)
    
    second_line = np.divide(second_line, 5)
    third_line = np.divide(third_line, 10)

    print('Finished example experiment')
    bandit = [5.412751265243876, 5.4778281339275825, 5.525198134975608, 5.38658040162747, 5.558336675471582, 5.498140374838126, 5.3897760116181335, 5.597195090537716, 5.523368069638705, 5.380369257755376, 5.56245964135598, 5.498308281202354, 5.435402986655783, 5.507656748965964, 5.394419187161896, 5.300965494979958, 5.428684803121478, 5.4391525208127955, 5.500003255679384, 5.431271210135738, 5.5654521355125315, 5.380577218734497, 5.490641365159842, 5.418484019226734, 5.425853130361867, 5.430268425293436, 5.399260332784894, 5.425440205782168, 5.6980645199037046, 5.450295052332275, 5.403409959777882, 5.553265337695963, 5.314751329250083, 5.38525026351177, 5.447440780587535, 5.3629358783717045, 5.480886500501842, 5.447637771420829, 5.633954770730964, 5.528425256464681, 5.55094489648924, 5.409842133527685, 5.432632035694864, 5.442636936210664, 5.438247138137831, 5.436951801837266, 5.406580172042959, 5.332157491607608, 5.587656008254113, 5.586327535310515, 5.558723438396797, 5.54131818454838, 5.28820929527524, 5.62977075269238, 5.484838249276647, 5.359062673858294, 5.4539239631776, 5.336022486812259, 5.323478168993665, 5.460499588763557, 5.412950169127858, 5.263784490461768, 5.370842545449385, 5.365354780627216, 5.590070134022492, 5.2698530163472155, 5.530223604427059, 5.464960995265785, 5.453751232435839, 5.296745948622613, 5.426559970098989, 5.562742660005341, 5.501800075735159, 5.3945948218760496, 5.473733728017079, 5.56394928571636, 5.620162494887425, 5.634346631697555, 5.559521329821447, 5.456251160995456, 5.394617752467545, 5.477658324255474, 5.549183287323986, 5.541552935207473, 5.302285150113759, 5.517302345537786, 5.461221139229094, 5.399248089060759, 5.455909024812645, 5.368443850297567, 5.425431279204042, 5.5128081901327635, 5.553060390328804, 5.538381462375159, 5.404009076301948, 5.4772342084414305, 5.613344147226892, 5.44312020534994, 5.486630338517518, 5.365694644195385]

    
    print("first line")
    print(first_line)
    print("second line")
    print(second_line)
    print("third line")
    print(third_line.tolist())
    
    label = []
    for i in range(100):
        label.append(i)

    fig = plt.figure()
    ax = plt.axes()
    plt.xlabel("np.floor(steps / 100)")
    plt.ylabel("reward")
    plt.plot(label, bandit, label='MultiArmedBanditrewards ')
    plt.plot(label, third_line, label="QLearning")
    plt.legend()
    plt.show()
    fig.savefig('2b.png', bbox_inches='tight')


def q2e():
    first_line = 0
    second_line = 0
    third_line = 0
    for i in range(10):
        env = gym.make('FrozenLake-v0')
        agent = MultiArmedBandit()
        action_values, rewards = agent.fit(env, 100000)
        if i == 0:
            first_line = rewards
            second_line = rewards
            third_line = rewards
        elif i > 0 and i < 5:
            second_line = np.add(second_line, rewards)
            third_line = np.add(third_line, rewards)
        else:
            third_line = np.add(third_line, rewards)
    
    second_line = np.divide(second_line, 5)
    third_line = np.divide(third_line, 10)

    print('Finished example experiment bandit')
    print("first line")
    print(first_line)
    print("second line")
    print(second_line)
    print("third line")
    print(third_line.tolist())
    bandit = third_line

    first_line = 0
    second_line = 0
    third_line = 0
    for i in range(10):
        env = gym.make('FrozenLake-v0')
        agent = QLearning()
        action_values, rewards = agent.fit(env, 100000)
        if i == 0:
            first_line = rewards
            second_line = rewards
            third_line = rewards
        elif i > 0 and i < 5:
            second_line = np.add(second_line, rewards)
            third_line = np.add(third_line, rewards)
        else:
            third_line = np.add(third_line, rewards)
    
    second_line = np.divide(second_line, 5)
    third_line = np.divide(third_line, 10)

    print('Finished example experiment Qlearning')
    print("first line")
    print(first_line)
    print("second line")
    print(second_line)
    print("third line")
    print(third_line.tolist())

    label = []
    for i in range(100):
        label.append(i)

    fig = plt.figure()
    ax = plt.axes()
    plt.xlabel("np.floor(steps / 100)")
    plt.ylabel("reward")
    plt.plot(label, bandit, label='MultiArmedBandit')#marker = 'o')
    plt.plot(label, third_line, label="Qlearngin")
    plt.legend()
    plt.show()
    fig.savefig('2e.png', bbox_inches='tight')


def q3a():
    first_line = 0
    second_line = 0
    third_line = 0
    for i in range(10):
        env = gym.make('FrozenLake-v0')
        agent = QLearning(epsilon=0.01, discount=0.95)
        action_values, rewards = agent.fit(env, 100000)
        if i == 0:
            first_line = rewards
            second_line = rewards
            third_line = rewards
        elif i > 0 and i < 5:
            second_line = np.add(second_line, rewards)
            third_line = np.add(third_line, rewards)
        else:
            third_line = np.add(third_line, rewards)
    
    second_line = np.divide(second_line, 5)
    third_line = np.divide(third_line, 10)

    print('Finished example experiment bandit')
    print("first line")
    print(first_line)
    print("second line")
    print(second_line)
    print("third line")
    print(third_line.tolist())
    e_001 = third_line

    first_line = 0
    second_line = 0
    third_line = 0
    for i in range(10):
        env = gym.make('FrozenLake-v0')
        agent = QLearning(epsilon=0.5, discount=0.95)
        action_values, rewards = agent.fit(env, 100000)
        if i == 0:
            first_line = rewards
            second_line = rewards
            third_line = rewards
        elif i > 0 and i < 5:
            second_line = np.add(second_line, rewards)
            third_line = np.add(third_line, rewards)
        else:
            third_line = np.add(third_line, rewards)
    
    second_line = np.divide(second_line, 5)
    third_line = np.divide(third_line, 10)

    print('Finished example experiment bandit')
    print("first line")
    print(first_line)
    print("second line")
    print(second_line)
    print("third line")
    print(third_line.tolist())

    label = []
    for i in range(100):
        label.append(i)

    fig = plt.figure()
    ax = plt.axes()
    plt.xlabel("np.floor(steps / 100)")
    plt.ylabel("reward")
    plt.plot(label, e_001, label='e = 0.01')#marker = 'o')
    plt.plot(label, third_line, label="e = 0.5")
    plt.legend()
    plt.show()
    fig.savefig('3a.png', bbox_inches='tight')

def q3c():
    first_line = 0
    second_line = 0
    third_line = 0
    for i in range(10):
        env = gym.make('FrozenLake-v0')
        agent = QLearning(epsilon=0.5, discount=0.95, adaptive=True)
        action_values, rewards = agent.fit(env, 100000)
        if i == 0:
            first_line = rewards
            second_line = rewards
            third_line = rewards
        elif i > 0 and i < 5:
            second_line = np.add(second_line, rewards)
            third_line = np.add(third_line, rewards)
        else:
            third_line = np.add(third_line, rewards)
    
    second_line = np.divide(second_line, 5)
    third_line = np.divide(third_line, 10)

    print('Finished example experiment bandit')
    print("first line")
    print(first_line)
    print("second line")
    print(second_line)
    print("third line")
    print(third_line)
    e_001 = [0.0046, 0.0071, 0.0075000000000000015, 0.0066, 0.007299999999999999, 0.008499999999999999, 0.0071, 0.0063, 0.009200000000000002, 0.0095, 0.01, 0.008100000000000001, 0.009999999999999998, 0.009699999999999997, 0.0079, 0.008, 0.0081, 0.009, 0.01, 0.008299999999999998, 0.008199999999999999, 0.007200000000000001, 0.0086, 0.0081, 0.01, 0.0104, 0.010099999999999998, 0.0112, 0.0082, 0.009899999999999999, 0.008799999999999999, 0.0115, 0.0091, 0.0097, 0.0098, 0.009999999999999998, 0.011, 0.010799999999999999, 0.011799999999999998, 0.009699999999999999, 0.011699999999999999, 0.0098, 0.011199999999999998, 0.008199999999999999, 0.0113, 0.0103, 0.011299999999999998, 0.011, 0.0116, 0.010100000000000001, 0.0093, 0.010299999999999998, 0.010199999999999999, 0.0112, 0.010399999999999998, 0.011, 0.0091, 0.010499999999999999, 0.010199999999999999, 0.0107, 0.011899999999999999, 0.0117, 0.0108, 0.0097, 0.0116, 0.010599999999999997, 0.0117, 0.0105, 0.011800000000000001, 0.0106, 0.01, 0.0108, 0.0109, 0.011399999999999999, 0.011000000000000001, 0.0126, 0.0127, 0.010499999999999999, 0.010099999999999998, 0.011899999999999999, 0.008100000000000001, 0.0116, 0.013599999999999998, 0.012499999999999999, 0.010799999999999999, 0.011199999999999998, 0.0115, 0.0116, 0.009999999999999998, 0.010599999999999998, 0.0118, 0.0134, 0.0114, 0.012, 0.0132, 0.0106, 0.011399999999999999, 0.012, 0.013000000000000001, 0.0115]


    e_05 = [0.0028000000000000004, 0.003799999999999999, 0.004000000000000001, 0.004600000000000002, 0.0041, 0.0044, 0.0050999999999999995, 0.004999999999999999, 0.0063, 0.006199999999999999, 0.005599999999999999, 0.0033, 0.0051, 0.0055000000000000005, 0.0058000000000000005, 0.0063, 0.0055, 0.005000000000000001, 0.005300000000000001, 0.0048000000000000004, 0.005599999999999999, 0.0062, 0.0039, 0.006100000000000001, 0.005299999999999999, 0.005, 0.006500000000000001, 0.004999999999999999, 0.005200000000000001, 0.0068000000000000005, 0.005599999999999999, 0.0055, 0.0058, 0.006, 0.0052, 0.0060999999999999995, 0.006199999999999999, 0.004699999999999999, 0.0062, 0.0060999999999999995, 0.006500000000000001, 0.0059, 0.0066, 0.0052, 0.005599999999999999, 0.004999999999999999, 0.005599999999999999, 0.006500000000000001, 0.005200000000000001, 0.005700000000000001, 0.005200000000000001, 0.0064, 0.0056, 0.004899999999999999, 0.0045, 0.006, 0.0061, 0.006900000000000001, 0.0057, 0.0060999999999999995, 0.0053, 0.005300000000000001, 0.0068000000000000005, 0.005800000000000001, 0.0059, 0.006199999999999999, 0.0055, 0.005299999999999999, 0.0064, 0.0058000000000000005, 0.0055, 0.0056, 0.006, 0.0058, 0.0059, 0.006, 0.0049, 0.0054, 0.0057, 0.0051, 0.0061, 0.0058, 0.005900000000000001, 0.0058, 0.0058, 0.0058, 0.0063, 0.0059, 0.0063, 0.0047, 0.0052, 0.004400000000000001, 0.0057, 0.006, 0.0055, 0.006, 0.0046, 0.0041, 0.0062, 0.0049]


    label = []
    for i in range(100):
        label.append(i)

    fig = plt.figure()
    ax = plt.axes()
    plt.xlabel("np.floor(steps / 100)")
    plt.ylabel("reward")
    plt.plot(label, e_001, label='e = 0.01')#marker = 'o')
    plt.plot(label, e_05, label="e = 0.5")
    plt.plot(label, third_line, label="e = 0.5 adp")
    plt.legend()
    plt.show()
    fig.savefig('3c.png', bbox_inches='tight')


if __name__ == "__main__":
    print('Starting example experiment')
    start = time.time()
    
    # q2a()
    # q2b()
    # q2e()
    # q3a()
    q3c()


    end = time.time()
    train_time = end - start
    print("train_time")
    print(train_time)