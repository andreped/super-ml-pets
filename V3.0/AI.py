import argparse
import time
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from Environment import Environment, TabularActor
from rudder import LessonBuffer
from rudder import RRLSTM as LSTM
import tqdm

parser = argparse.ArgumentParser(description='Rudder Demonstration')

parser.add_argument('--policy_learning', default="RUDDER", type=str, dest="pl", required=True,
                    help="The policy learning method. Valid options are \"RUDDER\", \"MC\" and \"TD\"",
                    choices=["RUDDER", "TD", "MC"])
update_rule = parser.parse_args().pl

if update_rule == "RUDDER":
    improve_policy = TabularActor.update_direct_q_estimation
    redistribute_reward = True
elif update_rule == "MC":
    improve_policy = TabularActor.update_monte_carlo
    redistribute_reward = False
elif update_rule == "TD":
    improve_policy = TabularActor.update_q_learning
    redistribute_reward = False
else:
    raise ValueError("Specify an update rule.")

rnd_seed = 1
torch.manual_seed(rnd_seed)
np.random.seed(rnd_seed)

lb_size = 2048
n_lstm = 16
max_time = 50
policy_lr = 0.1
lstm_lr = 1e-2
l2_regularization = 1e-6
avg_window = 750

env = Environment(avg_window=avg_window, transport_time=max_time)
state = env.reset()

reward_history = []

# Initialize LSTM Actor as well as the LessonBuffer.
eps_agent = TabularActor(env, lr=policy_lr)
if redistribute_reward:
    lesson_buffer = LessonBuffer(size=lb_size, max_time=max_time, n_features=env.get_state_shape()[-1])
    rudder_lstm = LSTM(state_input_size=8, n_actions=env.get_n_actions()[-1], buffer=lesson_buffer, n_units=n_lstm,
                       lstm_lr=lstm_lr, l2_regularization=l2_regularization, return_scaling=10,
                       lstm_batch_size=8, continuous_pred_factor=0.5)

start_time = time.time()
all_suboptimal_actions = []
episode = 0

print("Starting training using update rule: \"{}\"".format(update_rule) + "\n")
print("---------------------------------------------------------------")
print("Episode |    # poor | brand/ | % good decisions | runtime stats")
print("        | decisions | action |      goal: > 90% |              ")
pbar = tqdm.tqdm(desc="        |           |      n |                  | ", ncols=0)

while episode < 5000:
    episode += 1
    state = eps_agent.reset()
    done = False
    rewards = []
    states = [state]
    actions = []
    while not done:
        state, a, reward, done = eps_agent.act()
        actions.append(a)
        states.append(state)
        rewards.append(reward)
        if done:
            eps_agent.reset()
            states = np.stack(states)
            actions = np.array(actions)
            rewards = np.array(rewards)
            # If RUDDER is chosen
            if redistribute_reward:
                lesson_buffer.add(states=states, actions=actions, rewards=rewards)
                if lesson_buffer.different_returns_encountered() and lesson_buffer.full_enough():
                    # If RUDDER is run, the LSTM is trained after each episode until its loss is below a threshold.
                    # Samples will be drawn from the lessons buffer.
                    if episode % 25 == 0:
                        rudder_lstm.train(episode=episode)
                    # Then the LSTM is used to redistribute the reward.
                    rewards = rudder_lstm.redistribute_reward(states=np.expand_dims(states, 0),
                                                              actions=np.expand_dims(actions, 0))[0, :]
            # Train the policy, with the chosen learning method.
            improve_policy(eps_agent, states, actions, rewards)
            
            # Update the progressbar
            desc = f"{episode:7} |    {states[0, -2]}/{actions[0]} |"
            pbar.set_description(desc)
            pbar.update(1)
            pbar.refresh()

    if episode % 10 == 0:
        with open('ckptrew-'+episode, 'wb') as f:
            pickle.dump(rewards, f)
        with open('ckptstates-'+episode, 'wb') as f:
            pickle.dump(states, f)
        with open('ckptactions-'+episode, 'wb') as f:
            pickle.dump(actions, f)

print()
print(f"Done! (runtime: {time.time() - start_time}")
