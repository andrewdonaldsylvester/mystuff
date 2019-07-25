import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
import matplotlib.pyplot as plt

import tensorflow as tf

episodes = 100


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.997
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()


    def _build_model(self):
        # build the model
        model = Sequential()
        # create 4 input nodes with 24 hidden nodes on second layer
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        # create 24 hidden nodes on third layer
        model.add(Dense(24, activation='relu'))
        # have 2 output node with a step function
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    # remembers the state of the cartpole
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # gets a random number and sees if it is lower than the epsilon
    # if it is, the cartpole performs a random action
    # else, we try and predict what action to perform in that state
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
            self.model.fit(state, target, epochs=2, verbose=0)

        # decrease the epsilon by 0.3%
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    cartpoleAI = DQNAgent(state_size, action_size)
    batch_size = 32
    plot = []

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        done = False
        time = 0
        while not done:
            time += 1
            action = cartpoleAI.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            cartpoleAI.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                cartpoleAI.update_target_model()
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, episodes, time, cartpoleAI.epsilon))
                plot += [time]
                break

            # replay once there is enough "test runs" remembered
            if len(cartpoleAI.memory) > batch_size:
                cartpoleAI.replay(batch_size)

        with open('finalproject.txt', 'w') as fin:
            if time > 400:
                fin.write('Weights from episode {0} and score {1}: {2} \n'.format(e, time,cartpoleAI.target_model.get_weights()))



    plt.plot(plot)
    plt.show()
