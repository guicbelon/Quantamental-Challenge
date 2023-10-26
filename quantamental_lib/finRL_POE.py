import copy
import gym
import torch
import numpy as np
import torch.nn.functional as F

from tqdm.notebook import tqdm
from collections import deque

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from torch.optim import AdamW

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

"""From FinRL https://github.com/AI4Finance-LLC/FinRL/tree/master/finrl/env"""
import gym
import math
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding

from stable_baselines3.common.vec_env import DummyVecEnv
from pathlib import Path

try:
    import quantstats as qs
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        """QuantStats module not found, environment can't plot results and calculate indicadors.
        This module is not installed with FinRL. Install by running one of the options:
        pip install quantstats --upgrade --no-cache-dir
        conda install -c ranaroussi quantstats
        """
        )

class PortfolioOptimizationEnv(gym.Env):
    """A portfolio allocantion environment for OpenAI gym.

    This environment simulates the interactions between an agent and the financial market
    based on data provided by a dataframe. The dataframe contains the time series of
    features defined by the user (such as closing, high and low prices) and must have
    a time and a tic column with a list of datetimes and ticker symbols respectively.
    An example of dataframe is shown below::

            date        high            low             close           tic
        0   2020-12-23  0.157414        0.127420        0.136394        ADA-USD
        1   2020-12-23  34.381519       30.074295       31.097898       BNB-USD
        2   2020-12-23  24024.490234    22802.646484    23241.345703    BTC-USD
        3   2020-12-23  0.004735        0.003640        0.003768        DOGE-USD
        4   2020-12-23  637.122803      560.364258      583.714600      ETH-USD
        ... ...         ...             ...             ...             ...

    Based on this dataframe, the environment will create an observation space that can
    be a Dict or a Box. The Box observation space is a three-dimensional array of shape
    (f, n, t), where f is the number of features, n is the number of stocks in the
    portfolio and t is the user-defined time window. If the environment is created with
    the parameter return_last_action set to True, the observation space is a Dict with
    the following keys::

        {
        "state": three-dimensional Box (f, n, t) representing the time series,
        "last_action": one-dimensional Box (n+1,) representing the portfolio weights
        }

    Note that the action space of this environment is an one-dimensional Box with size
    n + 1 because the portfolio weights must contains the weights related to all the
    stocks in the portfolio and to the remaining cash.

    Attributes:
        action_space: Action space.
        observation_space: Observation space.
        episode_length: Number of timesteps of an episode.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df,
        initial_amount,
        order_df=True,
        return_last_action=False,
        normalize_df="by_previous_time",
        reward_scaling=1,
        comission_fee_model="trf",
        comission_fee_pct=0,
        features=["close", "high", "low"],
        valuation_feature="close",
        time_column="date",
        time_format="%Y-%m-%d",
        tic_column="tic",
        time_window=1,
        cwd="./",
        new_gym_api=False
    ):
        """Initializes environment's instance.

        Args:
            df: Dataframe with market information over a period of time.
            initial_amount: Initial amount of cash available to be invested.
            order_df: If True input dataframe is ordered by time.
            return_last_action: If True, observations also return the last performed
                action. Note that, in that case, the observation space is a Dict.
            normalize_df: Defines the normalization method applied to input dataframe.
                Possible values are "by_previous_time", "by_fist_time_window_value",
                "by_COLUMN_NAME" (where COLUMN_NAME must be changed to a real column
                name) and a custom function. If None no normalization is done.
            reward_scaling: A scaling factor to multiply the reward function. This
                factor can help training.
            comission_fee_model: Model used to simulate comission fee. Possible values
                are "trf" (for transaction remainder factor model) and "wvm" (for weights
                vector modifier model). If None, commission fees are not considered.
            comission_fee_pct: Percentage to be used in comission fee. It must be a value
                between 0 and 1.
            features: List of features to be considered in the observation space. The
                items of the list must be names of columns of the input dataframe.
            valuation_feature: Feature to be considered in the portfolio value calculation.
            time_column: Name of the dataframe's column that contain the datetimes that
                index the dataframe.
            time_format: Formatting string of time column.
            tic_name: Name of the dataframe's column that contain ticker symbols.
            time_window: Size of time window.
            cwd: Local repository in which resulting graphs will be saved.
            new_gym_api: If True, the environment will use the new gym api standard for
                step and reset methods.
        """
        # super(StockEnv, self).__init__()
        # money = 10 , scope = 1
        self._time_window = time_window
        self._time_index = time_window - 1
        self._time_column = time_column
        self._time_format = time_format
        self._tic_column = tic_column
        self._df = df
        self._initial_amount = initial_amount
        self._return_last_action = return_last_action
        self._reward_scaling = reward_scaling
        self._comission_fee_pct = comission_fee_pct
        self._comission_fee_model = comission_fee_model
        self._features = features
        self._valuation_feature = valuation_feature
        self._cwd = Path(cwd)
        self._new_gym_api = new_gym_api

        # price variation
        self._df_price_variation = None

        # preprocess data
        self._preprocess_data(order_df, normalize_df)

        # dims and spaces
        self._tic_list = self._df[self._tic_column].unique()
        self._stock_dim = len(self._tic_list)
        action_space = 1 + self._stock_dim

        # sort datetimes and define episode length
        self._sorted_times = sorted(set(self._df[time_column]))
        self.episode_length = len(self._sorted_times) - time_window + 1

        # define action space
        self.action_space = spaces.Box(low=0, high=1, shape=(action_space,))

        # define observation state
        if self._return_last_action:
            # if  last action must be returned, a dict observation
            # is defined
            self.observation_space = spaces.Dict({
                "state": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(
                        len(self._features),
                        self._stock_dim,
                        self._time_window
                    )
                ),
                "last_action": spaces.Box(
                    low=0, high=1, shape=(action_space,)
                )
            })
        else:
            # if information about last action is not relevant,
            # a 3D observation space is defined
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(
                    len(self._features),
                    self._stock_dim,
                    self._time_window
                ),
            )

        self._reset_memory()

        self._portfolio_value = self._initial_amount
        self._terminal = False

    def step(self, actions):
        """Performs a simulation step.

        Args:
            actions: An unidimensional array containing the new portfolio
                weights.

        Note:
            If the environment was created with "return_last_action" set to
            True, the next state returned will be a Dict. If it's set to False,
            the next state will be a Box. You can check the observation state
            through the attribute "observation_space".

        Returns:
            If "new_gym_api" is set to True, the following tuple is returned:
            (state, reward, terminal, truncated, info). If it's set to False,
            the following tuple is returned: (state, reward, terminal, info).

            state: Next simulation state.
            reward: Reward related to the last performed action.
            terminal: If True, the environment is in a terminal state.
            truncated: If True, the environment has passed it's simulation
                time limit. Currently, it's always False.
            info: A dictionary containing informations about the last state.
        """
        self._terminal = self._time_index >= len(self._sorted_times) - 1

        if self._terminal:
            metrics_df = pd.DataFrame(
                {"date": self._date_memory,
                 "returns": self._portfolio_return_memory,
                 "rewards": self._portfolio_reward_memory,
                 "portfolio_values": self._asset_memory["final"]}
            )
            metrics_df.set_index("date", inplace=True)

            print("=================================")
            print("Initial portfolio value:{}".format(self._asset_memory['final'][0]))
            print("Final portfolio value: {}".format(self._portfolio_value))
            print("Final accumulative portfolio value: {}".format(self._portfolio_value / self._asset_memory['final'][0]))
            print("Maximum DrawDown: {}".format(qs.stats.max_drawdown(metrics_df["portfolio_values"])))
            print("Sharpe ratio: {}".format(qs.stats.sharpe(metrics_df["returns"])))
            print("=================================")

            if self._new_gym_api:
                return self._state, self._reward, self._terminal, False, self._info
            return self._state, self._reward, self._terminal, self._info

        else:
            # transform action to numpy array (if it's a list)
            actions = np.array(actions, dtype=np.float32)

            # if necessary, normalize weights
            if math.isclose(np.sum(actions), 1, abs_tol=1e-6) and np.min(actions) >= 0:
                weights = actions
            else:
                weights = self._softmax_normalization(actions)

            # save initial portfolio weights for this time step
            self._actions_memory.append(weights)

            # get last step final weights and portfolio_value
            last_weights = self._final_weights[-1]

            # load next state
            self._time_index += 1
            self._state, self._info = self._get_state_and_info_from_time_index(self._time_index)

            # if using weights vector modifier, we need to modify weights vector
            if self._comission_fee_model == "wvm":
                delta_weights = weights - last_weights
                delta_assets = delta_weights[1:] # disconsider
                # calculate fees considering weights modification
                fees = np.sum(np.abs(delta_assets * self._portfolio_value))
                if fees > weights[0] * self._portfolio_value:
                    weights = last_weights
                    # maybe add negative reward
                else:
                    portfolio = weights * self._portfolio_value
                    portfolio[0] -= fees
                    self._portfolio_value = np.sum(portfolio) # new portfolio value
                    weights = portfolio / self._portfolio_value # new weights
            elif self._comission_fee_model == "trf":
                last_mu = 1
                mu = 1 - 2 * self._comission_fee_pct + self._comission_fee_pct ** 2
                while abs(mu - last_mu) > 1e-10:
                    last_mu = mu
                    mu = (1 - self._comission_fee_pct * weights[0] -
                          (2 * self._comission_fee_pct - self._comission_fee_pct ** 2) *
                          np.sum(np.maximum(last_weights[1:] - mu * weights[1:], 0))) / (1 - self._comission_fee_pct * weights[0])
                self._info["trf_mu"] = mu
                self._portfolio_value = mu * self._portfolio_value

            # save initial portfolio value of this time step
            self._asset_memory["initial"].append(self._portfolio_value)

            # time passes and time variation changes the portfolio distribution
            portfolio = self._portfolio_value * (weights * self._price_variation)

            # calculate new portfolio value and weights
            self._portfolio_value = np.sum(portfolio)
            weights = portfolio / self._portfolio_value

            # save final portfolio value and weights of this time step
            self._asset_memory["final"].append(self._portfolio_value)
            self._final_weights.append(weights)

            # save date memory
            self._date_memory.append(self._info["end_time"])

            # define portfolio return
            rate_of_return = self._asset_memory["final"][-1] / self._asset_memory["final"][-2]
            portfolio_return = rate_of_return - 1
            portfolio_reward = np.log(rate_of_return)

            # save portfolio return memory
            self._portfolio_return_memory.append(portfolio_return)
            self._portfolio_reward_memory.append(portfolio_reward)

            # Define portfolio return
            self._reward = portfolio_reward
            self._reward = self._reward * self._reward_scaling

        if self._new_gym_api:
            return self._state, self._reward, self._terminal, False, self._info
        return self._state, self._reward, self._terminal, self._info

    def reset(self):
        """Resets the environment and returns it to its initial state (the
        fist date of the dataframe).

        Note:
            If the environment was created with "return_last_action" set to
            True, the initial state will be a Dict. If it's set to False,
            the initial state will be a Box. You can check the observation
            state through the attribute "observation_space".

        Returns:
            If "new_gym_api" is set to True, the following tuple is returned:
            (state, info). If it's set to False, only the initial state is
            returned.

            state: Initial state.
            info: Initial state info.
        """
        # time_index must start a little bit in the future to implement lookback
        self._time_index = self._time_window - 1
        self._reset_memory()

        self._state, self._info = self._get_state_and_info_from_time_index(self._time_index)
        self._portfolio_value = self._initial_amount
        self._terminal = False

        if self._new_gym_api:
            return self._state, self._info
        return self._state

    def _get_state_and_info_from_time_index(self, time_index):
        """Gets state and information given a time index. It also updates "data"
        attribute with information about the current simulation step.

        Args:
            time_index: An integer that represents the index of a specific datetime.
                The initial datetime of the dataframe is given by 0.

        Note:
            If the environment was created with "return_last_action" set to
            True, the returned state will be a Dict. If it's set to False,
            the returned state will be a Box. You can check the observation
            state through the attribute "observation_space".

        Returns:
            A tuple with the following form: (state, info).

            state: The state of the current time index. It can be a Box or a Dict.
            info: A dictionary with some informations about the current simulation
                step. The dict has the following keys::

                {
                "tics": List of ticker symbols,
                "start_time": Start time of current time window,
                "start_time_index": Index of start time of current time window,
                "end_time": End time of current time window,
                "end_time_index": Index of end time of current time window,
                "data": Data related to the current time window,
                "price_variation": Price variation of current time step
                }
        """
        # returns state in form (channels, tics, timesteps)
        end_time = self._sorted_times[time_index]
        start_time = self._sorted_times[time_index - (self._time_window - 1 )]

        # define data to be used in this time step
        self._data = self._df[
            (self._df[self._time_column] >= start_time) &
            (self._df[self._time_column] <= end_time)
        ][[self._time_column, self._tic_column] + self._features]

        # define price variation of this time_step
        self._price_variation = self._df_price_variation[
                self._df_price_variation[self._time_column] == end_time
            ][self._valuation_feature].to_numpy()
        self._price_variation = np.insert(self._price_variation, 0, 1)

        # define state to be returned
        state = None
        for tic in self._tic_list:
            tic_data = self._data[self._data[self._tic_column] == tic]
            tic_data = tic_data[self._features].to_numpy().T
            tic_data = tic_data[..., np.newaxis]
            state = tic_data if state is None else np.append(state, tic_data, axis=2)
        state = state.transpose((0, 2, 1))
        info = {
            "tics": self._tic_list,
            "start_time": start_time,
            "start_time_index": time_index - (self._time_window - 1),
            "end_time": end_time,
            "end_time_index": time_index,
            "data": self._data,
            "price_variation": self._price_variation
        }
        return self._standardize_state(state), info

    def render(self, mode="human"):
        """Renders the environment.

        Returns:
            Observation of current simulation step.
        """
        return self._state

    def _softmax_normalization(self, actions):
        """Normalizes the action vector using softmax function.

        Returns:
            Normalized action vector (portfolio vector).
        """
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        softmax_output = numerator / denominator
        return softmax_output

    def enumerate_portfolio(self):
        """Enumerates the current porfolio by showing the ticker symbols
        of all the investments considered in the portfolio.
        """
        print("Index: 0. Tic: Cash")
        for index, tic in enumerate(self._tic_list):
            print("Index: {}. Tic: {}".format(index + 1, tic))

    def _preprocess_data(self, order, normalize):
        """Orders and normalizes the environment's dataframe.

        Args:
            order: If true, the dataframe will be ordered by ticker list
                and datetime.
            normalize: Defines the normalization method applied to the dataframe.
                Possible values are "by_previous_time", "by_fist_time_window_value",
                "by_COLUMN_NAME" (where COLUMN_NAME must be changed to a real column
                name) and a custom function. If None no normalization is done.
        """
        # order time dataframe by tic and time
        if order:
            self._df = self._df.sort_values(by=[self._tic_column, self._time_column])
        # defining price variation after ordering dataframe
        self._df_price_variation = self._temporal_variation_df()
        # apply normalization
        if normalize:
            self._normalize_dataframe(normalize)
        # transform str to datetime
        self._df[self._time_column] = pd.to_datetime(self._df[self._time_column])
        self._df_price_variation[self._time_column] = pd.to_datetime(self._df_price_variation[self._time_column])
        # transform numeric variables to float32 (compatibility with pytorch)
        self._df[self._features] = self._df[self._features].astype("float32")
        self._df_price_variation[self._features] = self._df_price_variation[self._features].astype("float32")

    def _reset_memory(self):
        """Resets the environment's memory."""
        date_time = self._sorted_times[self._time_index]
        # memorize portfolio value each step
        self._asset_memory = {
            "initial" : [self._initial_amount],
            "final" : [self._initial_amount]
        }
        # memorize portfolio return and reward each step
        self._portfolio_return_memory = [0]
        self._portfolio_reward_memory = [0]
        # initial action: all money is allocated in cash
        self._actions_memory = [np.array([1] + [0] * self._stock_dim, dtype=np.float32)]
        # memorize portfolio weights at the ending of time step
        self._final_weights = [np.array([1] + [0] * self._stock_dim, dtype=np.float32)]
        # memorize datetimes
        self._date_memory = [date_time]

    def _standardize_state(self, state):
        """Standardize the state given the observation space. If "return_last_action"
        is set to False, a three-dimensional box is returned. If it's set to True, a
        dictionary is returned. The dictionary follows the standard below::

            {
            "state": Three-dimensional box representing the current state,
            "last_action": One-dimensional box representing the last action
            }
        """
        last_action = self._actions_memory[-1]
        if self._return_last_action:
            return { "state": state, "last_action": last_action }
        else:
            return state

    def _normalize_dataframe(self, normalize):
        """"Normalizes the environment's dataframe.

        Args:
            normalize: Defines the normalization method applied to the dataframe.
                Possible values are "by_previous_time", "by_fist_time_window_value",
                "by_COLUMN_NAME" (where COLUMN_NAME must be changed to a real column
                name) and a custom function. If None no normalization is done.

        Note:
            If a custom function is used in the normalization, it must have an
            argument representing the environment's dataframe.
        """
        if type(normalize) == str:
            if normalize == "by_fist_time_window_value":
                self._df = self._temporal_variation_df(self._time_window - 1)
            elif normalize == "by_previous_time":
                self._df = self._temporal_variation_df()
            elif normalize.startswith("by_"):
                normalizer_column = normalize[3:]
                for column in self._features:
                    self._df[column] = self._df[column] / self._df[normalizer_column]
        elif callable(normalize):
            self._df = normalize(self._df)


    def _temporal_variation_df(self, periods=1):
        """Calculates the temporal variation dataframe. For each feature, this
        dataframe contains the rate of the current feature's value and the last
        feature's value given a period. It's used to normalize the dataframe.

        Args:
            periods: Periods (in time indexes) to calculate temporal variation.

        Returns:
            Temporal variation dataframe.
        """
        df_temporal_variation = self._df.copy()
        prev_columns = []
        for column in self._features:
            prev_column = "prev_{}".format(column)
            prev_columns.append(prev_column)
            df_temporal_variation[prev_column] = df_temporal_variation.groupby(self._tic_column)[column].shift(periods=periods)
            df_temporal_variation[column] = df_temporal_variation[column] / df_temporal_variation[prev_column]
        df_temporal_variation = df_temporal_variation.drop(columns=prev_columns).fillna(1).reset_index(drop=True)
        return df_temporal_variation

    def _seed(self, seed=None):
        """Seeds the sources of randomness of this environment to guarantee
        reproducibility.

        Args:
            seed: Seed value to be applied.

        Returns:
            Seed value applied.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self, env_number=1):
        """Generates an environment compatible with Stable Baselines 3. The
        generated environment is a vectorized version of the current one.

        Returns:
            A tuple with the generated environment and an initial observation.
        """
        e = DummyVecEnv([lambda: self] * env_number)
        obs = e.reset()
        return e, obs


class GradientPolicy(nn.Module):
    def __init__(self):
        """DDPG policy network initializer."""
        super().__init__()

        self.sequential = nn.Sequential(
            nn.Conv2d( in_channels=23, out_channels=2, kernel_size=(1, 3)),
            nn.Softmax(),
            nn.Conv2d(in_channels=2, out_channels=20, kernel_size=(1, 48)),
            nn.ReLU()
        )

        self.final_convolution = nn.Conv2d(in_channels=21, out_channels=1, kernel_size=(1, 1))

        self.softmax = nn.Sequential(
            nn.Softmax(dim=-1)
        )

    def _fix_tensor(self,tensor, float_value=0.0):
        mask_nan = torch.isnan(tensor)
        tensor = torch.where(mask_nan, torch.tensor(float_value), tensor)
        mask_inf = torch.isinf(tensor)
        tensor = torch.where(mask_inf, torch.tensor(float_value), tensor)
        return tensor


    def mu(self, observation, last_action):
        """Defines a most favorable action of this policy given input x.

        Args:
          observation: environment observation .
          last_action: Last action performed by agent.

        Returns:
          Most favorable action.
        """

        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation).to(device)
        if isinstance(last_action, np.ndarray):
            last_action = torch.from_numpy(last_action).to(device)

        last_stocks, cash_bias = self._process_last_action(last_action)
        
        observation = self._fix_tensor(observation)
        last_stocks = self._fix_tensor(last_stocks)
        cash_bias = self._fix_tensor(cash_bias)

        output = self.sequential(observation) # shape [N, PORTFOLIO_SIZE + 1, 19, 1]
        output = self._fix_tensor(output)
        
        
        output = torch.cat([output, last_stocks], dim=1) # shape [N, 21, PORTFOLIO_SIZE, 1]
        output = self._fix_tensor(output)
        output = self.final_convolution(output) # shape [N, 1, PORTFOLIO_SIZE, 1]
        output = self._fix_tensor(output)
        output = torch.cat([output, cash_bias], dim=2) # shape [N, 1, PORTFOLIO_SIZE + 1, 1]
        output = self._fix_tensor(output)

        # output shape must be [N, features] = [1, PORTFOLIO_SIZE + 1], being N batch size (1)
        # and size the number of features (weights vector).
        output = torch.squeeze(output, 3)
        output = self._fix_tensor(output)

        #print('out4',output)
        output = torch.squeeze(output, 1) # shape [N, PORTFOLIO_SIZE + 1]
        
        output = self._fix_tensor(output)

        output = self.softmax(output)
        #print('out',output)

        return output

    def forward(self, observation, last_action):
        """Policy network's forward propagation.

        Args:
          observation: environment observation (dictionary).
          epsilon: exploration noise to be applied.

        Returns:
          Action to be taken (numpy array).
        """
        mu = self.mu(observation, last_action)
        action = mu.cpu().detach().numpy().squeeze()
        return action

    def _process_last_action(self, last_action):
        batch_size = last_action.shape[0]
        stocks = last_action.shape[1] - 1
        last_stocks = last_action[:, 1:].reshape((batch_size, 1, stocks, 1))
        cash_bias = last_action[:, 0].reshape((batch_size, 1, 1, 1))
        return last_stocks, cash_bias
    
class PVM:
    def __init__(self, capacity, portfolio_size):
        """Initializes portfolio vector memory.

        Args:
          capacity: Max capacity of memory.
        """
        # initially, memory will have the same actions
        self.capacity = capacity
        self.portfolio_size = portfolio_size
        self.reset()

    def reset(self):
        self.memory = [np.array([1] + [0] * (self.portfolio_size), dtype=np.float32)] * (self.capacity + 1)
        self.index = 0 # initial index to retrieve data

    def retrieve(self):
        last_action = self.memory[self.index]
        self.index = 0 if self.index == self.capacity else self.index + 1
        return last_action

    def add(self, action):
        self.memory[self.index] = action



class ReplayBuffer:
  def __init__(self, capacity):
    """Initializes replay buffer.

    Args:
      capacity: Max capacity of buffer.
    """
    self.buffer = deque(maxlen=capacity)

  def __len__(self):
    """Represents the size of the buffer

    Returns:
      Size of the buffer.
    """
    return len(self.buffer)

  def append(self, experience):
    """Append experience to buffer. When buffer is full, it pops
       an old experience.

    Args:
      experience: experience to be saved.
    """
    self.buffer.append(experience)

  def sample(self):
    """Sample from replay buffer. All data from replay buffer is
    returned and the buffer is cleared.

    Returns:
      Sample of batch_size size.
    """
    buffer = list(self.buffer)
    self.buffer.clear()
    return buffer
  

class RLDataset(IterableDataset):
  def __init__(self, buffer):
    """Initializes reinforcement learning dataset.

    Args:
        buffer: replay buffer to become iterable dataset.

    Note:
        It's a subclass of pytorch's IterableDataset,
        check https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
    """
    self.buffer = buffer

  def __iter__(self):
    """Iterates over RLDataset.

    Returns:
      Every experience of a sample from replay buffer.
    """
    for experience in self.buffer.sample():
        yield experience


def polyak_average(net, target_net, tau=0.01):
  """Applies polyak average to incrementally update target net.

  Args:
    net: trained neural network.
    target_net: target neural network.
    tau: update rate.
  """
  for qp, tp in zip(net.parameters(), target_net.parameters()):
    tp.data.copy_(tau * qp.data + (1 - tau) * tp.data)

class PG:
    def __init__(self,
                 env,
                 portfolio_size,
                 batch_size=100,
                 lr=1e-3,
                 optimizer=AdamW,
                 tau=0.05):
        """Initializes Policy Gradient for portfolio optimization.

          Args:
            env: environment.
            batch_size: batch size to train neural network.
            lr: policy neural network learning rate.
            optim: Optimizer of neural network.
            tau: update rate in Polyak averaging.
        """
        # environment
        self.env = env

        # neural networks
        self.policy = GradientPolicy().to(device)
        self.target_policy = copy.deepcopy(self.policy)
        self.optimizer = optimizer(self.policy.parameters(), lr=lr)
        self.tau = tau

        # replay buffer and portfolio vector memory
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(capacity=batch_size)
        self.pvm = PVM(self.env.episode_length, portfolio_size= portfolio_size)

        # dataset and dataloader
        dataset = RLDataset(self.buffer)
        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True)

    def train(self, episodes=100):
        """Training sequence

        Args:
            episodes: Number of episodes to simulate
        """
        for i in tqdm(range(1, episodes + 1)):
            obs = self.env.reset() # observation
            self.pvm.reset() # reset portfolio vector memory
            done = False

            while not done:
                # define last_action and action and update portfolio vector memory
                last_action = self.pvm.retrieve()
                obs_batch = np.expand_dims(obs, axis=0)
                last_action_batch = np.expand_dims(last_action, axis=0)
                action = self.policy(obs_batch, last_action_batch)
                self.pvm.add(action)

                # run simulation step
                next_obs, reward, done, info = self.env.step(action)

                # add experience to replay buffer
                exp = (obs, last_action, info["price_variation"], info["trf_mu"])
                self.buffer.append(exp)

                # update policy networks
                if len(self.buffer) == self.batch_size:
                    self._gradient_ascent()

                obs = next_obs

            # gradient ascent with episode remaining buffer data
            self._gradient_ascent()



    def _gradient_ascent(self):
        # update target neural network
        polyak_average(self.policy, self.target_policy, tau=self.tau)

        # get batch data from dataloader
        obs, last_actions, price_variations, trf_mu = next(iter(self.dataloader))
        obs = obs.to(device)
        last_actions = last_actions.to(device)
        price_variations = price_variations.to(device)
        trf_mu = trf_mu.unsqueeze(1).to(device)

        # define policy loss (negative for gradient ascent)
        mu = self.policy.mu(obs, last_actions)
        policy_loss = - torch.mean(torch.log(torch.sum(mu * price_variations * trf_mu, dim=1)))

        # update policy network
        self.policy.zero_grad()
        policy_loss.backward()
        self.optimizer.step()


