import numpy as np
import time
import datetime
import gym
import os
from keras.models import Sequential, load_model
from keras import layers
import matplotlib.pyplot as plt

class DQNAgent():
    # left, stand, right
    actionNum = 3
    episode = 500 #1600
    stepsPerEpisode = 500
    gamma = 0.9
    batchSize = 10 #32

    env = gym.make('MountainCar-v0')
    env._max_episode_step = 1000
    successCnt = 0
    dataMemory = []
    maxScsRate = 0.0
    maxScsCnt = 0
    rtnModel = None
    startTime = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    plotX = [] # training steps
    plotY = [] # success count, position >= 0.5
    plotAcc = []
    plotScsCnt = []

    def __init__(self):
        self.model = Sequential()
        self.model.add(layers.Dense(100, input_shape=(2,), activation='relu'))
        self.model.add(layers.Dense(self.actionNum))
        self.model.compile(optimizer='adam', loss='mse')
        
    # train the NN model for output action-value
    def trainAgent(self, epsilon=0.13):

        for i in range(self.episode):

            # Get initial state
            state = np.reshape(self.env.reset(), (1, 2))
            
            if i != 0:
                # Update action-value to get training data
                train_x, train_y = self.getTrainingData(self.dataMemory)
                
                self.model.fit(train_x, train_y,
                                batch_size=self.batchSize,
                                validation_split=0.1,
                                verbose=0, epochs=100)
            print("Episode - {}, Success count: {}".format(i, self.successCnt))
            
            # Clear the memory
            self.dataMemory.clear()
            localSuccessCnt = 0
            reached = False
            
            # Get training set
            for step in range(self.stepsPerEpisode):

                # Refresh enviornment
                # self.env.render()

                # Get action-value
                Q = self.model.predict(state)[0]
                
                if np.random.rand() < epsilon:
                    action = np.random.randint(0, self.actionNum, size=1)[0]
                    tmp = Q[np.argmax(Q)]
                    Q[np.argmax(Q)] = Q[action]
                    Q[action] = tmp
                else:
                    # Take the best action 
                    action = np.argmax(Q)

                observation, reward, _,  _ = self.env.step(action)
                
                reward += 1 + abs(observation[0]) + observation[0] * 3

                if observation[0] >= 0.5:
                    reached = True
                    for index, (Q, st, rwd) in enumerate(self.dataMemory):
                        rwd += 10
                    reward += 10
                    localSuccessCnt += 1
                    self.successCnt += 1

                # Save the next state and the reward to the previous action
                self.dataMemory.append((Q, state, reward))
                state = np.reshape(observation, (1, 2))
                self.plotX.append(step + i * self.episode)
                self.plotY.append(self.successCnt)

            if (reached):
                print("Saving checkpoint {}...".format(i))
                self.saveCheckpoint(episode=i)
                curScsRate = self.test(self.model)
                print("CurAcc: {0:.3f}, maxAcc: {1:.3f}".format(curScsRate, self.maxScsRate))
                if curScsRate > self.maxScsRate:
                    self.maxScsRate = curScsRate
                    self.rtnModel = load_model("./model/" + self.startTime + "/checkpoint" + str(i))
                    print("Updating the best returned model...")
            self.plotAcc.append(self.maxScsRate)
            self.plotScsCnt.append(self.successCnt)

        print("Finish training with episode {}, batchSize {}".format(self.episode, self.batchSize))
        if self.rtnModel != None: self.saveModel(self.rtnModel)
        return self.rtnModel

    def saveModel(self, model):
        print("Saving model...")
        st = self.startTime
        modelDir = "./model/"
        model.save(modelDir + "model_" + st)
        print("Finish saving model!")

    def saveCheckpoint(self, episode, dirName=startTime):
        modelDir = "./model/" + dirName
        if not os.path.exists(modelDir): os.makedirs(modelDir)
        self.model.save(modelDir + "/" + "checkpoint" + str(episode))

    def getTrainingData(self, memory):
        train_x = np.zeros((len(memory) - 1, 2))
        train_y = np.zeros((len(memory) - 1, 3))
        for index, (Q, state, reward) in enumerate(memory):
            if index == len(memory) - 1 :
                continue        

            nextQ, _, _ = memory[index + 1]

            Q = reward + self.gamma * nextQ
            
            train_x[index] = state
            train_y[index] = Q

        return train_x, train_y

    def test(self, model, rounds=500):
        env = gym.make('MountainCar-v0')
        env._max_episode_step = 1000
        
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
        reachTopRate = scsCnt / rounds
        return reachTopRate

    def plotLine(self, x=plotX, y=plotY):
        fig = plt.figure()
        plt.plot(x, y)
        plt.xlabel('step', fontsize=18)
        plt.ylabel('Success count (pos >= 0.5)', fontsize=16)
        plt.show()
        fig.savefig("./img/" + str(self.episode) + 'ep_' + str(self.batchSize) +'batch' + '.jpg')
    
    def plotTwoScale(self, s1=plotAcc, s2=plotScsCnt, x=list(range(episode))):
        fig, ax1 = plt.subplots()
        ax1.plot(x, s1, 'b-')
        ax1.set_xlabel('episode (s)')
        ax1.set_ylabel('Best accuracy', color='b')
        ax1.tick_params('y', colors='b')

        ax2 = ax1.twinx()
        ax2.plot(x, s2, 'r-')
        ax2.set_ylabel('Success count', color='r')
        ax2.tick_params('y', colors='r')

        fig.tight_layout()
        fig.savefig("./img/" + str(self.episode) + 'ep_' + str(self.batchSize) +'batch-twoScale.jpg')
        plt.show()
                