import numpy as np
import sap


class Environment:

    def __init__(self, avg_window, transport_time=50):
        self.list = []

        # sap
        self.sim = sap.SAP()

        self.total_wins = 0
        self.total_losses = 0
        self.total_draws = 0 

    def reset(self):
        self.total_wins += self.sim.wins
        self.total_losses += self.sim.losses
        self.total_draws += self.sim.draws 

        self.sim = sap.SAP()
        return np.array(self.sim.get_scaled_state(), dtype=np.int32)

    def step(self, action):
        reward = -1000
        
        result = self.sim.step()

        reward += self.sim.score

        done = self.sim.isGameOver() or not result

        newstate = np.array(self.sim.get_scaled_state(),
                            dtype=np.int32), reward, done
        return newstate

    # Used for Q-table initialization, provides shape.
    def get_state_max_values(self):
        return [1]

    def get_state_shape(self):
        return [2]

    def get_n_actions(self):
        return [69]


class TabularActor:

    def __init__(self, env, lr):
        self.q_table = np.empty(shape=(env.get_n_actions() + env.get_state_max_values()))
        self.q_table[:] = np.nan
        self.q_table[1, :] = 10  # Optimistic initialization
        self.q_table[..., 0] = 10  # Optimistic initialization
        self.q_table[..., -1] = 0
        self.env = env
        self.state = env.reset()
        self.lr = lr

    def reset(self):
        self.state = self.env.reset()
        return self.state

    def act(self):
        state = self.state
        q_s = self.q_table[(slice(0, None),) + tuple(state)]
        # Be greedy
        if len(np.unique(q_s)) > 1:
            a = np.nanargmax(q_s)
        else:
            a = np.random.choice(2)

        # Explore with 5% chance.
        if np.random.random() < 0.10:
            a = np.random.choice(2)
        if self.state[-1] > 0:
            a = 1

        # Step forward in the environment.
        self.state, reward, done = self.env.step(a)
        return self.state, a, reward, done

    # Three different update rules to change the policy:

    # "RUDDER learning"
    # Direct Q-Value estimation, using redistributed reward of RUDDER
    def update_direct_q_estimation(self, states, actions, rewards):
        for i in range(actions.shape[0]):
            self.q_table[tuple([actions[i]] + states[i, :].tolist())] += self.lr * (
                    rewards[i] - self.q_table[tuple([actions[i]] + states[i, :].tolist())])

    # Q-Learning update
    def update_q_learning(self, states, actions, rewards):
        indices = np.concatenate([np.expand_dims(np.array(actions.tolist() + [0]), 1), states], axis=1)
        maxq = [self.q_table[tuple([slice(0, None), *indices[i, 1:]])] for i in range(indices.shape[0])]
        maxq = np.nanmax(np.array(maxq), axis=1)
        for t in range(actions.shape[0]):
            self.q_table[tuple(indices[t, :])] = (1 - self.lr) * self.q_table[tuple(indices[t, :])] + self.lr * (
                    rewards[t] + maxq[t + 1])

    # Monte-Carlo control update
    def update_monte_carlo(self, states, actions, rewards):
        gt = rewards[::-1].cumsum()[::-1]
        for i in range(actions.shape[0]):
            self.q_table[tuple([actions[i]] + states[i, :].tolist())] += self.lr * (
                    gt[i] - self.q_table[tuple([actions[i]] + states[i, :].tolist())])
