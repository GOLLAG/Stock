import numpy as np
import pandas as pd
import gym
from gym import spaces

class StockTradingEnv(gym.Env):
    def __init__(self, data):
        super(StockTradingEnv, self).__init__()

        # Load and preprocess data
        self.data = data.copy()

        # Convert 'Date' to a numeric format (Unix timestamp)
        self.data['Date'] = pd.to_datetime(self.data['Date'], format='%m/%d/%Y')  # Specify format if known
        self.data['Date'] = self.data['Date'].map(pd.Timestamp.timestamp)  # Convert to Unix timestamp

        # Remove commas and convert 'Open', 'High', 'Low', 'Close' to numeric
        for column in ['Open', 'High', 'Low', 'Close']:
            self.data[column] = self.data[column].str.replace(',', '').astype(float)
        
        self.current_step = 0
        self.max_steps = len(data) - 1

        # Action space: Buy, Hold, Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: stock prices (Open, High, Low, Close) and portfolio value
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(len(data.columns) + 1,), dtype=np.float32)
        
        self.initial_balance = 10000
        self.balance = float(self.initial_balance)
        self.stock_held = 0.0
        self.stock_price = 0.0
        self.total_assets = float(self.initial_balance)

    def reset(self):
        self.current_step = 0
        self.balance = float(self.initial_balance)
        self.stock_held = 0.0
        self.stock_price = 0.0
        self.total_assets = float(self.initial_balance)
        return self._get_observation()

    def _get_observation(self):
        return np.concatenate([
            self.data.iloc[self.current_step].values.astype(np.float32),
            [self.total_assets]
        ])

    def step(self, action):
        self.current_step += 1
        if self.current_step >= len(self.data):
            self.current_step = len(self.data) - 1
        
        prev_balance = self.balance
        self.stock_price = float(self.data.iloc[self.current_step]['Close'])
        
        if action == 0:  # Buy
            if self.balance >= self.stock_price:
                self.stock_held += self.balance / self.stock_price
                self.balance = 0.0
        elif action == 1:  # Hold
            pass
        elif action == 2:  # Sell
            self.balance += self.stock_held * self.stock_price
            self.stock_held = 0.0
        
        self.total_assets = self.balance + self.stock_held * self.stock_price
        reward = self.total_assets - prev_balance
        
        done = self.current_step == self.max_steps
        
        return self._get_observation(), reward, done, {}

    def render(self, mode='human'):
        print(f'Step: {self.current_step}, Balance: {self.balance}, Stock Held: {self.stock_held}, Total Assets: {self.total_assets}')

# Example usage
data = pd.read_csv('historical_stock_prices.csv')

# Initialize the environment
env = StockTradingEnv(data)

# Example of taking a step in the environment
obs = env.reset()
print(f'Initial Observation: {obs}')

action = env.action_space.sample()  # Take a random action
obs, reward, done, _ = env.step(action)
print(f'Next Observation: {obs}, Reward: {reward}, Done: {done}')
env.render()
