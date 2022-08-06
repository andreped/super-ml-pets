"""
Main application to train RL model in simulated environment
"""

import sys, os
from collections import deque
import time
import random
import pickle, csv

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tqdm import tqdm

import sap
from utils import Data, get_qs, save_logs
from models import agent

def train(env, replay_memory, model, target_model, done):
    learning_rate = 0.7 # Learning rate
    discount_factor = 0.618

    MIN_REPLAY_SIZE = 1000
    if len(replay_memory) < MIN_REPLAY_SIZE:
        return

    batch_size = 64 * 2
    mini_batch = random.sample(replay_memory, batch_size)
    current_states = np.array([transition[0] for transition in mini_batch])
    current_qs_list = model.predict(current_states, verbose=0)
    new_current_states = np.array([transition[3] for transition in mini_batch])
    future_qs_list = target_model.predict(new_current_states, verbose=0)

    X = []
    Y = []
    for index, (observation, action, reward, new_observation, done) in enumerate(mini_batch):
        if not done:
            max_future_q = reward + discount_factor * np.max(future_qs_list[index])
        else:
            max_future_q = reward

        current_qs = current_qs_list[index]
        current_qs[action] = (1 - learning_rate) * current_qs[action] + learning_rate * max_future_q

        X.append(observation)
        Y.append(current_qs)

    model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True)

def main(start_episode, verbose_step):
    epsilon = 1 # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start
    max_epsilon = 1 # You can't explore more than 100% of the time
    min_epsilon = 0.01 # At a minimum, we'll always explore 1% of the time
    decay = 0.01

    data = Data()

    # 1. Initialize the Target and Main models
    # Main Model (updated every 4 steps)
    if not finetune:
        model = agent((50, ), 69)
        start_episode = 1
    else:
        model = keras.models.load_model('ckpt/ckpt-' + str(start_episode))
        data.total_wins = 0
        data.total_draws = 0
        data.total_losses = 0
        #start_episode = 12000

        with open('past_teams_bin', 'rb') as f:
            data.past_teams = pickle.load(f)

        print("loaded")

    
    # Target Model (updated every 100 steps)
    target_model = agent((50, ), 69)
    target_model.set_weights(model.get_weights())

    replay_memory = deque(maxlen=50_000)

    # X = states, y = actions
    X = []
    y = []
    target_update_counter = 0
    steps_to_update_target_model = 0

    try:
        for episode in tqdm(range(start_episode, start_episode + train_episodes), "Iter:"):
            total_training_rewards = 0
            env = sap.SAP(data)
            observation = env.get_scaled_state()
            done = False
            while not done:
                steps_to_update_target_model += 1

                random_number = np.random.rand()
                # 2. Explore using the Epsilon Greedy Exploration Strategy
                if random_number <= epsilon:
                    # Explore
                    action = [0]*69  # @TODO: Is this OK to do in Python 3.8?
                    action[np.random.randint(0, 69)] = 1
                else:
                    # Exploit best known action
                    # model dims are (batch, env.observation_space.n)
                    encoded = observation
                    encoded_reshaped = encoded.reshape([1, encoded.shape[0]])
                    action = model.predict(encoded_reshaped, verbose=0).flatten()

                env.step(action)

                action = np.argmax(action)

                new_observation = env.get_scaled_state()
                reward = env.score 
                done = env.isGameOver()
                info = None
                replay_memory.append([observation, action, reward, new_observation, done])

                # 3. Update the Main Network using the Bellman Equation
                if steps_to_update_target_model % 4 == 0 or done:
                    train(env, replay_memory, model, target_model, done)

                observation = new_observation
                total_training_rewards += reward

                if done:
                    if episode % verbose_step == 0:
                        print('After n steps = {} : Total training rewards: {} '.format(episode, total_training_rewards))
                    total_training_rewards += 1

                    if steps_to_update_target_model >= 100:
                        # print('Copying main network weights to the target network weights')
                        target_model.set_weights(model.get_weights())
                        steps_to_update_target_model = 0
                    break

            data.total_wins += env.wins
            data.total_losses += env.losses
            data.total_draws += env.draws
            
            if episode % verbose_step == 0:
                print("stats: ", data.total_wins, "/", data.total_draws, "/", data.total_losses)

            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)

            if episode % 100 == 0:
                save_logs(data)
                model.save('ckpt/ckpt-'+str(episode))

    except KeyboardInterrupt:
        print('Interrupted')
        save_logs(data)
        model.save('ckpt/ckpt-'+episode)
        print("stats: ", data.total_wins, "/", data.total_draws, "/", data.total_losses)
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

    
    save_logs(data)
    model.save('ckpt/ckpt-'+str(episode))

if __name__ == '__main__':
    # An episode a full game
    start_episode = 41800
    train_episodes = 1000000
    test_episodes = 100
    finetune = True  # whether to finetune or not
    verbose_step = 10

    main(start_episode=start_episode, verbose_step=verbose_step)
