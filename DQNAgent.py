import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.001 # min exploration rate
        self.epsilon_decay = 0.999 # decay of exploration rate
        self.learning_rate = 0.001 # learning rate
        self.model = self._build_model() # builds a neural network
        self.target_model = self._build_model() # builds a copy
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
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    # remembers the state of the cartpole and the next frame
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # gets a random number and sees if it is lower than the epsilon
    # if it is, the cartpole performs a random action
    # else, we try and predict what action to perform in that state
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns best action

    def replay(self, batch_size):
        # takes 32 random frames from memory
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            # gives a prediction using the current model
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
            self.model.fit(state, target, epochs=1, verbose=0)

        # decrease the epsilon by 0.3% until it reaches 0.001
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay