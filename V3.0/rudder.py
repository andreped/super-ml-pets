import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from nn import LSTMLayer
from torch.autograd import Variable
from torch.nn import MSELoss as MSELoss


def to_one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot


class RRLSTM(nn.Module):
    def __init__(self, state_input_size, n_actions, buffer, n_units, lstm_lr, l2_regularization,
                 return_scaling, lstm_batch_size=128, continuous_pred_factor=0.5):
        super(RRLSTM, self).__init__()
        self.buffer = buffer
        self.return_scaling = return_scaling
        self.lstm_batch_size = lstm_batch_size
        self.continuous_pred_factor = continuous_pred_factor
        self.n_actions = n_actions
        # Forget gate and output gate are deactivated as used in the Atari games, see Appendix S4.2.1
        self.lstm = LSTMLayer(in_features=state_input_size + n_actions, out_features=n_units,
                              w_ci=(lambda *args, **kwargs: nn.init.normal_(mean=0, std=0.1, *args, **kwargs), False),
                              w_ig=(False, lambda *args, **kwargs: nn.init.normal_(mean=0, std=0.1, *args, **kwargs)),
                              w_og=False,
                              b_ci=lambda *args, **kwargs: nn.init.normal_(mean=0, *args, **kwargs),
                              b_ig=lambda *args, **kwargs: nn.init.normal_(mean=-3, *args, **kwargs),
                              b_og=False,
                              a_out=lambda x: x
                              )
        self.linear = nn.Linear(n_units, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lstm_lr, weight_decay=l2_regularization)
        self.lstm_updates = 0

    def forward(self, input):
        states, actions = input
        # Prepare input features
        repaired = states[:, :, 0:1]
        transport_cond = states[:, :, 1:3]
        brands = to_one_hot(states[:, :, 3], 4)
        time = states[:, :, 4:] / states.shape[1]
        states = torch.cat([repaired, transport_cond, brands, time], 2)
        actions = to_one_hot(actions, self.n_actions)
        actions = torch.cat((actions, torch.zeros((actions.shape[0], 1, self.n_actions))), 1)
        input = torch.cat((states, actions), 2)
        # Run the lstm
        lstm_out = self.lstm.forward(input, return_all_seq_pos=True)
        return self.linear(lstm_out[0])

    def redistribute_reward(self, states, actions):
        # Prepare LSTM inputs
        states_var = Variable(torch.FloatTensor(states)).detach()
        delta_states = torch.cat([states_var[:, 0:1, :], states_var[:, 1:, :] - states_var[:, :-1, :]], dim=1)
        actions_var = Variable(torch.FloatTensor(actions)).detach()
        # Calculate LSTM predictions
        lstm_out = self.forward([delta_states, actions_var])
        pred_g0 = torch.cat([torch.zeros_like(lstm_out[:, 0:1, :]), lstm_out], dim=1)[:, :-1, :]

        # Difference of predictions of two consecutive timesteps.
        redistributed_reward = pred_g0[:, 1:, 0] - pred_g0[:, :-1, 0]

        # Scale reward back up as LSTM targets have been scaled.
        new_reward = redistributed_reward * self.return_scaling
        return new_reward

    # Trains the LSTM until -on average- the main loss is below 0.25.
    def train(self, episode):

        i = 0
        loss_average = 0.3
        mse_loss = MSELoss(reduction="none")
        while loss_average > 0.15:
            i += 1
            self.lstm_updates += 1
            self.optimizer.zero_grad()

            # Get samples from the lesson buffer and prepare them.
            states, actions, rewards, lenght = self.buffer.sample(self.lstm_batch_size)
            lenght = lenght[:, 0]
            states_var = Variable(torch.FloatTensor(states)).detach()
            actions_var = Variable(torch.FloatTensor(actions)).detach()
            rewards_var = Variable(torch.FloatTensor(rewards)).detach()

            # Scale the returns as they might have high / low values.
            returns = torch.sum(rewards_var, 1, keepdim=True) / self.return_scaling

            # Calculate differences of states
            delta_states = torch.cat([states_var[:, 0:1, :], states_var[:, 1:, :] - states_var[:, :-1, :]], dim=1)

            # Run the LSTM
            lstm_out = self.forward([delta_states, actions_var])
            predicted_G0 = lstm_out.squeeze()

            # Loss calculations
            all_timestep_loss = mse_loss(predicted_G0, returns.repeat(1, predicted_G0.size(1)))

            # Loss at any position in the sequence
            aux_loss = self.continuous_pred_factor * all_timestep_loss.mean()

            # LSTM is mainly trained on getting the final prediction of g0 right.
            main_loss = all_timestep_loss[range(self.lstm_batch_size), lenght[:] - 1].mean()

            # LSTM update and loss tracking
            lstm_loss = main_loss + aux_loss
            lstm_loss.backward()
            loss_np = lstm_loss.data.numpy()
            main_loss_np = main_loss.data.numpy()
            loss_average -= 0.01 * (loss_average - main_loss_np)
            if main_loss_np > loss_average * 2:
                loss_average = loss_np
            self.optimizer.step()


class LessonBuffer:
    def __init__(self, size, max_time, n_features):
        self.size = size
        # Samples, time, features
        self.states_buffer = np.empty(shape=(size, max_time + 1, n_features))
        self.actions_buffer = np.empty(shape=(size, max_time))
        self.rewards_buffer = np.empty(shape=(size, max_time))
        self.lens_buffer = np.empty(shape=(size, 1), dtype=np.int32)
        self.next_spot_to_add = 0
        self.buffer_is_full = False
        self.samples_since_last_training = 0

    # LSTM training does only make sense, if there are sequences in the buffer which have different returns.
    # LSTM could otherwise learn to ignore the input and just use the bias units.
    def different_returns_encountered(self):
        if self.buffer_is_full:
            return np.unique(self.rewards_buffer[..., -1]).shape[0] > 1
        else:
            return np.unique(self.rewards_buffer[:self.next_spot_to_add, -1]).shape[0] > 1

    # We only train if 64 samples are played by a random policy
    def full_enough(self):
        return self.buffer_is_full or self.next_spot_to_add > 256

    # Add a new episode to the buffer
    def add(self, states, actions, rewards):
        traj_length = states.shape[0]
        next_ind = self.next_spot_to_add
        self.next_spot_to_add = self.next_spot_to_add + 1
        if self.next_spot_to_add >= self.size:
            self.buffer_is_full = True
        self.next_spot_to_add = self.next_spot_to_add % self.size
        self.states_buffer[next_ind, :traj_length] = states.squeeze()
        self.states_buffer[next_ind, traj_length:] = 0
        self.actions_buffer[next_ind, :traj_length - 1] = actions
        self.actions_buffer[next_ind, traj_length:] = 0
        self.rewards_buffer[next_ind, :traj_length - 1] = rewards
        self.rewards_buffer[next_ind, traj_length:] = 0
        self.lens_buffer[next_ind] = traj_length

    # Choose <batch_size> samples uniformly at random and return them.
    def sample(self, batch_size):
        self.samples_since_last_training = 0
        if self.buffer_is_full:
            indices = np.random.randint(0, self.size, batch_size)
        else:
            indices = np.random.randint(0, self.next_spot_to_add, batch_size)
        return (self.states_buffer[indices, :, :], self.actions_buffer[indices, :],
                self.rewards_buffer[indices, :], self.lens_buffer[indices, :])


def nograd(t):
    return t.detach()
