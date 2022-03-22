import numpy as np


class WatchRepairEnvironment:

    def __init__(self, avg_window, transport_time=50):
        self.list = []

        # Brands and their properties
        self.n_brands = 4
        self.brand_appearance_probability = np.array([0.25, 0.25, 0.25, 0.25])
        self.brand_sale_price = np.array([18, 28, 31, 59])

        self.average_brand_related_transport_cost = np.array([0.5, 2.5, 4.5, 18])
        self.brand_related_transport_cost_variance = np.array([1.5, 1.5, 1.5, 1.5])

        self.average_brand_repair_price = np.array([1, 4, 5, 24.5])
        self.brand_repair_variance = np.array([2, 2, 2, 1])

        # Non brand-related costs detectable by patterns in the state
        self.n_transport_conditions = 2
        self.transport_time = transport_time
        self.transport_cost = np.array([0.1, 7])

        self.transport_condition_probability = np.array([0.1, 0.05])

        # Instance variables which will be part of the state
        self.brand = -1
        self.repaired = None
        self.time = None
        self.transport_condition = None

        # For tracking performance
        # Hardcoded in this setup, only used for performance evaluation
        self.optimal_actions = np.array([1, 0, 0, 1])
        self.optimal_choices = 0
        self.optimal_actions_list = []
        self.avg_window = avg_window

    def reset(self):
        # Randomly choose a brand which the agent faces in this episode.
        self.brand = np.random.choice(self.n_brands, size=1, p=self.brand_appearance_probability)[0]
        self.repaired = 0
        self.time = 0
        self.transport_condition = np.zeros(shape=(self.n_transport_conditions,))
        return np.array([self.repaired] + self.transport_condition.tolist() + [self.brand, self.time], dtype=np.int32)

    def step(self, action):
        reward = 0
        if self.time == 0:
            # Check if the repair action was executed.
            self.repaired = action == 0
            if self.repaired:
                # If repair was chosen, immediately charge the repair cost
                repair_price = np.random.normal(self.average_brand_repair_price[self.brand],
                                                self.brand_repair_variance[self.brand])
                reward -= repair_price

            # Performance tracking: tracks the number of good decisions for the whole training.
            if action == self.optimal_actions[self.brand]:
                self.optimal_choices += 1
            # Update moving-mean window of latest optimal actions
            self.optimal_actions_list.append(action == self.optimal_actions[self.brand])
            if len(self.optimal_actions_list) > self.avg_window:
                del self.optimal_actions_list[0]

        self.time += 1
        done = self.time == self.transport_time

        # Generate transport events which increase delivery costs.
        transport_cond = self.transport_condition
        transport_cond += np.array(
            [np.random.random() < self.transport_condition_probability[i] for i in range(self.n_transport_conditions)])
        self.transport_condition = transport_cond

        if done and self.repaired:
            # Calculate the final reward by adding costs and sales price.
            transport_cost = np.sum(transport_cond * self.transport_cost)
            brand_related_transport_costs = np.random.normal(self.average_brand_related_transport_cost[self.brand],
                                                             self.brand_related_transport_cost_variance[self.brand])
            brand_sales_price = self.brand_sale_price[self.brand]
            reward += brand_sales_price - transport_cost - brand_related_transport_costs
            self.list.append(transport_cost)
        newstate = np.array([self.repaired] + transport_cond.tolist() + [self.brand, self.time],
                            dtype=np.int32), reward, done
        return newstate

    # Used for Q-table initialization, provides shape.
    def get_state_max_values(self):
        return [2] + np.repeat(self.transport_time, self.n_transport_conditions).tolist() + [self.n_brands,
                                                                                             self.transport_time + 1]

    def get_state_shape(self):
        return [3 + self.n_transport_conditions]

    def get_n_actions(self):
        return [2]


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
