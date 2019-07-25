import gym
import numpy as np
import matplotlib.pyplot as plt
from DQNAgent import *


env = gym.make('CartPole-v1') # initializing the game environment
state_size = env.observation_space.shape[0] # how many inputs the NN should take in
action_size = env.action_space.n # how many outputs the NN should give out
cartpoleAI = DQNAgent(state_size, action_size) # generating an agent to play the game with the right amount of inputs and outputs


episodes = 100 # how many episodes to play
batch_size = 32 # how much memory the AI needs to have before it starts learning from past episodes


fin = open('finalproject.txt', 'a') # opening a file to save good weights
scores = [] # making a list to hold the scores


for e in range(episodes):
    state = env.reset() # generate a new game state for each episode
    state = np.reshape(state, [1, state_size]) # formatting to 2d array

    done = False # used to tell when to stop episode
    time = 0 # counts how many timesteps the episode has been running, used for score

    while not done:
        time += 1 # increment time

        action = cartpoleAI.act(state) # determine which action to take

        next_state, reward, done, _ = env.step(action) # take the action and get the results of it
        next_state = np.reshape(next_state, [1, state_size]) # formatting to 2d array

        cartpoleAI.remember(state, action, reward, next_state, done) # storing the previous state in the memory

        state = next_state # updates state variable to the current state

        if done:
            cartpoleAI.update_target_model() # set the current model to the target model

            print("episode: {}/{}, score: {}, e: {:.2}".format(e, episodes, time, cartpoleAI.epsilon)) # displaying results from the episode

            scores += [time] # adding the score to the graph

            break # move to next episode

        if len(cartpoleAI.memory) > batch_size: # checks if there is enough data to train with
            cartpoleAI.replay(batch_size) # trains AI


        if time > 400: # writes the weights to a file if the score is high enough
            fin.write('Weights from episode {0} and score {1}: {2} \n'.format(e, time,cartpoleAI.target_model.get_weights()))



plt.plot(scores) # plots graph of scores
plt.show()
