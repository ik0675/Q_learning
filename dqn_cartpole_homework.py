import gymnasium as gym
import math
import random
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#Initialize the Environment and make an Initial Random Step
env = gym.make('CartPole-v1')
observation, info = env.reset(seed=42)
action = env.action_space.sample() 
observation, reward, terminated, truncated, info = env.step(action)

#Constants
BATCH_SIZE = 128    # Numbers of Samples fed into the nerual network during trainig at once
GAMMA = 0.99        # The decaying factor in the bellman function. Still remember the accumulated *discounted* return?
TAU = 0.005         # Update rate of the duplicate network
LR = 1e-4           # Learning rate of your Q - network

# The probability of choosing a random action will start at EPS_START and will decay exponentially towards EPS_END. EPS_DECAY controls the rate of the decay.
PROB_EPS_START = 0.9  # EPS_START is the starting value of epsilon
PROB_EPS_END = 0.05  # EPS_END is the final value of epsilon
RATE_EPS_DECAY = 1000 # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        # 4 X 2 (input 4, output 2)
        #TODO : Define the layer of your Q network here. Think about the shapes when you define it.
        #What is the shape of the input? What should be the shape of the output?
        self.first_layer = nn.Linear(n_observations, BATCH_SIZE)
        self.second_layer = nn.Linear(BATCH_SIZE, BATCH_SIZE)
        self.final_layer = nn.Linear(BATCH_SIZE, n_actions)

    def forward(self, x):
        #TODO : Define how the network should process your input to produce an output
        #Should return a tensor
        x = F.relu(self.first_layer(x))
        x = F.relu(self.second_layer(x))
        
        return self.final_layer(x)
    
#Creating to instances of the Q-network.
#Policy net is trained online directly by loss function
#Target network updates slower and provides a more stable target
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_taken = 0

def select_action(state):
    #TODO : Implement an epsilon-greedy policy that
    #Picks an random action with a small posibility
    #and acts according to the Q values otherwise
    
    
    if True:
        global steps_taken
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


episode_durations = []
steps_done = 0


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    #Obtain indices for all non-final states
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                        batch.next_state)), device=device, dtype=torch.bool)
    #The next states of all the non-final states
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    #Gather output of the Q-network
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    next_state_vales = torch.zeros(BATCH_SIZE, device=device)
    
    with torch.no_grad():
        #TODO : Calcualte the observed the Q value (Q_observed = immediate reward + gamma * max(Q(s_t+1)))
        #HINT : Use the target net for estimation of the next state
        next_state_vales[non_final_mask] = target_net(non_final_next_states).max(1).values
    expected_q_values = (next_state_vales * GAMMA) + reward_batch

    #TODO : Pick an appropiate loss function and calculate the loss
    #TODO : Name your calculated loss "loss"
    l1_loss = nn.SmoothL1Loss()
    loss = l1_loss(state_action_values, expected_q_values.unsqueeze(1))
    
    print("here")

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


if torch.cuda.is_available():
    num_episodes = 500
else:
    num_episodes = 50

for i_episode in range(num_episodes):
    # Initialize the environment and get it's state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    total_reward = 0
    num_steps = 0
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)
        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            break
    if i_episode % 10 == 0:
        print(i_episode)

env.close()

TestEnv = gym.make("CartPole-v1", render_mode="human")
observation, info = TestEnv.reset(seed=42)

end_count = 0
for _ in range(1000):
    state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    action = select_action(state) # this is where you would insert your policy
    observation, reward, terminated, truncated, _ = TestEnv.step(action.item())

    if terminated or truncated:
        end_count += 1
        observation, info = TestEnv.reset()

TestEnv.close()
print(end_count)