import numpy as np
import gym
import time
from model import *
from keras.models import load_model
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--train',
                        default=True,
                        help='input T/F to continue training the model')
    parser.add_argument('--modelPath',
                        default=None,
                        help='input the path to the existed model')
    parser.add_argument('--plot',
                        default=False,
                        help='input T\F to generate some training process graph')                        
    args = parser.parse_args()

    env = gym.make('MountainCar-v0')
    env._max_episode_steps = 1000
    model = None
    state = np.reshape(env.reset(), (1, 2))
    done = False

    if args.train == True:
        agent = DQNAgent()
        model = agent.trainAgent()
        print("Model accuracy: {0:.3f}".format(agent.maxScsRate))
        
        if args.plot == True:
            agent.plotLine()
            agent.plotTwoScale(agent.plotAcc, agent.plotScsCnt)
        if model == None: 
            print("Train model failed...")
            return
    else:
        if args.modelPath == None:
            print("Please input the path to the existed model by args")
        else: 
            model = load_model(args.modelPath)
            acc = test(model)
            print("Model accuracy: {0:.3f} ".format(acc))
            if acc >= 0.8 and args.plot == True: plotModelAction(model)

    for _ in range(1000):
        env.render()

        action = np.argmax(model.predict(state))
        nextState, _, _, _ = env.step(action)

        state = np.reshape(nextState, (1, 2))
        if nextState[0] >= 0.5:
            print("REACHED THE TOP!!!")
            done = True
            break

    if done == False: print("Didn't reach the top.")
    time.sleep(3)
    env.close() 

def test(model, rounds=500):
    env = gym.make('MountainCar-v0')
    env._max_episode_steps = 500
    scsCnt = 0
    for rd in range(rounds):
        done = False
        state = np.reshape(env.reset(), (1, 2))
        for _ in range(500):
            action = np.argmax(model.predict(state))
            nextState, _, _, _ = env.step(action)

            state = np.reshape(nextState, (1, 2))
            if nextState[0] >= 0.5:
                # REACHED THE TOP
                scsCnt += 1
                done = True
                break

    env.close()
    print("Success count: {}".format(scsCnt))
    reachTopRate = scsCnt / rounds
    return reachTopRate

def plotModelAction(model):
    color = {0:"red", 1:"blue", 2:"yellow"}
    pos = np.arange(-1.2, 0.6, 0.01)
    speed = np.arange(-0.07, 0.07, 0.001)
    fig = plt.figure()
    for x in pos:
        for y in speed:
            state = np.reshape([x, y], (1, 2))
            action = np.argmax(model.predict(state))
            plt.scatter(x, y, marker='o', color=color[action])
    plt.xlabel('position')
    plt.ylabel('velocity')
    plt.show()
    fig.savefig('model.jpg')

if __name__ == "__main__":
    main()