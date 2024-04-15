import gymnasium as gym
import math
import random
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
import numpy as np

# Create the environment
env = gym.make('CartPole-v1')

observation, info = env.reset(seed=42)

action = env.action_space.sample()  # this is where you would insert your policy
observation, reward, terminated, truncated, info = env.step(action)

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# Get number of actions from gym action space (Right or Left) # 2
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class PolicyNet(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(PolicyNet, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        # x = F.softmax(x)
        x = F.softmax(x, dim=0)
        return x


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 1)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


p_net = PolicyNet(n_observations, n_actions).to(device)
# v_net = DQN(n_observations, n_actions).to(device)
v_net = DQN(n_observations, n_actions).to(device)

p_optimizer = optim.AdamW(p_net.parameters(), lr=LR, amsgrad=True)
v_optimizer = optim.AdamW(v_net.parameters(), lr=LR, amsgrad=True)

memory = ReplayMemory(10000)


def select_action(state):
    state = torch.tensor([[env.action_space.sample()]],
                 device=device, dtype=torch.long)
    probs = p_net(state).cpu()
    m = Categorical(probs)
    action = m.sample()
    return action.item(), m.log_prob(action)


# episode_durations = []
# steps_done = 0


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                        batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = v_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = v_net(
            non_final_next_states).max(1)[0]
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    critic_loss = criterion(state_action_values,
                            expected_state_action_values.unsqueeze(1))

    probs = p_net(state_batch)
    m = Categorical(probs)
    action_log_probs = m.log_prob(action_batch)
    advantage = expected_state_action_values - state_action_values.detach()
    actor_loss = -(advantage * action_log_probs).mean()

    # Total loss
    total_loss = critic_loss + actor_loss

    # Update the network
    p_optimizer.zero_grad()
    v_optimizer.zero_grad()
    total_loss.backward()
    v_optimizer.step()
    v_optimizer.step()

    return total_loss


if torch.cuda.is_available():
    num_episodes = 500
else:
    num_episodes = 50

# writer = SummaryWriter()

# device = torch.device('cpu')


for i_episode in range(num_episodes):
    # Initialize the environment and get it's state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32,
                         device=device).unsqueeze(0)
    total_reward = 0
    num_steps = 0
    # saved_log_probs = []
    # rewards = []

    for t in count():
        action, log_prob = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action)
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(
                observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        optimize_model()
        
        # Move to the next state
        state = next_state

        # rewards.append(reward)
        # saved_log_probs.append(log_prob)
        
        total_reward += reward.item()
        num_steps += 1
        
        if done:
            episode_durations.append(t + 1)
            break

    # returns = deque(maxlen=500)
    # n_steps = len(rewards)

    # Show the average episode reward every 10 episodes
    if i_episode % 10 == 0:
        print(i_episode)

env.close()

TestEnv = gym.make("CartPole-v1", render_mode="human")
observation, info = TestEnv.reset(seed=42)

end_count = 0

for _ in range(1000):
    state = torch.tensor(observation, dtype=torch.float32,
                         device=device).unsqueeze(0)
    # this is where you would insert your policy
    action, _ = select_action(state)
    observation, reward, terminated, truncated, _ = TestEnv.step(action)

    if terminated or truncated:
        end_count += 1
        observation, info = TestEnv.reset()

TestEnv.close()

print("End Count:", end_count)
