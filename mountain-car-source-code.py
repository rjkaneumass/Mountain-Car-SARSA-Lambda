import math
import numpy as np
import random
import matplotlib.pyplot as plt

class Mountain_Car:
    def __init__(self):
        self.state = (-0.5, 0)
        self.actions = (0, 1, 2)
        self.epsilon = 0.1
        self.totalReward = 0

    def getNewState(self, state1, action):
        new_velocity = state1[1] + (0.001 * action) + (0.0025 * math.cos((3 * state1[0])))
        new_position = state1[0] + new_velocity
        if(new_position < -1.2):
            new_position = -1.2
            new_velocity = 0
        elif(new_position > 0.5):
            new_position = 0.5
            new_velocity = 0
        return (new_position, new_velocity)
    
    def getNewAction(self, q_outputs, epsilon):
        max = 0
        i = 0
        x = 0
        choice = random.random()
        if (choice <= epsilon):
            choice1 = random.randint(0, 2)
            return self.actions[choice1]
        else:
            index = np.argmax(q_outputs)
            return self.actions[index]

    def getReward(self, newState):
        if(newState[0] == 0.5):
            return 0
        else:
            return -1
        
    def reset(self):
        self.state = (-0.5, 0)
        self.totalReward = 0

    def makeMove(self):
        rewardsForPlot = np.zeros(300)
        allTrials = np.zeros((100, 300))
        finalResult = np.zeros(300)
        lowerStandardError = np.zeros(300)
        higherStandardError = np.zeros(300)
        episodes = 300
        trials = 100
        attempt = 0
        trialnumber = 0
        options1 = np.array([0, 0, 0])
        options2 = np.array([0, 0, 0])
        fourier = FourierBasis()
        while(trialnumber < trials):
            while(attempt < episodes):
                while (self.state[0] < 0.5):
                    choice = random.random()
                    if (choice <= self.epsilon):
                        choice = random.randrange(0, 2)
                        new_state = self.getNewState(self.state, self.actions[choice])
                        self.totalReward += self.getReward(new_state)
                        choice1 = random.randrange(0, 2)
                        fourier.update(new_state, self.actions[choice1], self.getReward(new_state), self.state, self.actions[choice])
                        self.state = new_state
                    else:
                        options1[0] = fourier.q_function(self.state, self.actions[0])
                        options1[1] = fourier.q_function(self.state, self.actions[1])
                        options1[2] = fourier.q_function(self.state, self.actions[2])
                        first_action = self.getNewAction(options1, self.epsilon)
                        new_state = self.getNewState(self.state, self.actions[first_action])
                        options2[0] = fourier.q_function(new_state, self.actions[0])
                        options2[1] = fourier.q_function(new_state, self.actions[1])
                        options2[2] = fourier.q_function(new_state, self.actions[2])
                        fourier.update(new_state, self.getNewAction(options2, self.epsilon), self.getReward(new_state), self.state, first_action)
                        self.totalReward += self.getReward(new_state)
                        self.state = new_state
                        options1 = np.array([0, 0, 0])
                        options2 = np.array([0, 0, 0])
                rewardsForPlot[attempt] = self.totalReward
                self.reset()
                attempt += 1
            allTrials[trialnumber] = rewardsForPlot
            rewardsForPlot = np.zeros(300)
            trialnumber += 1
            fourier.reset()
            attempt = 0
        episodecounter = 0
        trialcounter = 0
        total = 0
        total1 = 0
        standardErrorCounter = 0
        standardErrorCounterT = 0
        while(episodecounter < episodes):
            while(trialcounter < trials):
                total += allTrials[trialcounter][episodecounter]
                trialcounter += 1
            pusher = total/trials
            finalResult[episodecounter]=pusher
            total = 0
            trialcounter = 0
            episodecounter += 1
        while(standardErrorCounter < episodes):
            while(standardErrorCounterT < trials):
                total1 += (allTrials[standardErrorCounterT][standardErrorCounter] - finalResult[standardErrorCounter])**2
                standardErrorCounterT += 1
            pusher = ((total1/trials)**0.5)
            lowerStandardError[standardErrorCounter] = finalResult[standardErrorCounter] - (2 * pusher)
            higherStandardError[standardErrorCounter] = finalResult[standardErrorCounter] + (2 * pusher)
            standardErrorCounterT = 0
            total1 = 0
            standardErrorCounter += 1

        plt.plot(finalResult, label = "average reward")
        plt.plot(lowerStandardError, label = "Lower Standard Error")
        plt.plot(higherStandardError, label = "Higher Standard Error")
        plt.ylabel('Total Reward')
        plt.xlabel('Episode #')
        plt.title('Learning Output for Mountain_Car (averaged over 100 trials)')
        plt.legend()
        plt.show()




class FourierBasis():
    def __init__(self):
        self.weights = np.array([(0, 0, 0), (0, 0, 0), (0, 0, 0)], float) #array of tuples
        self.eligibility_trace = np.array([(0, 0, 0), (0, 0, 0), (0, 0, 0)], float)#array of tuples
        self.lambd = 0.92
        self.alpha = 0.00001

    def reset(self):
        self.weights = np.array([(0, 0, 0), (0, 0, 0), (0, 0, 0)], float) #array of tuples
        self.eligibility_trace = np.array([(0, 0, 0), (0, 0, 0), (0, 0, 0)], float)#array of tuples
    
    def phi(self, state, action_vector):
        result = np.cos((np.pi) * action_vector * state[0])
        return result

    def q_function(self, state, action):
        action_weights = self.weights[action]
        result = np.dot(action_weights, self.phi(state, action_weights))
        return result
    
    def update(self, next_state, next_action, reward, state, action):
        first_result = reward + self.q_function(next_state, next_action) - self.q_function(state, action)
        self.eligibility_trace[action] = self.lambd * self.eligibility_trace[action] + self.phi(state, self.weights[action])
        self.weights[action] += (self.alpha * first_result) * self.eligibility_trace[action]



mountaincar1 = Mountain_Car()
mountaincar1.makeMove()





        

    
    

    






