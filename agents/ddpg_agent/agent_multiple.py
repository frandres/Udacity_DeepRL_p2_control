import numpy as np
import random
from collections import namedtuple

import copy

#from .models import ActorNetwork,CriticNetwork
# from .models_baseline_diff_arch import Actor,Critic
from .models_baseline_bnorm import Actor,Critic
import torch
import torch.optim as optim
from torch import nn

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 64         # default minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
UPDATE_EVERY = 20       # how often to update the network
LEARN_FOR =10
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def annealing_generator(start: float,
                        end: float,
                        factor: float):
    decreasing = start > end

    eps = start
    while True:
        yield eps
        f = max if decreasing else min
        eps = f(end, factor*eps)


class Agent():
    '''
    DDPG Agent for solving the navegation system project.

    Skeleton adapted from Udacity exercise sample code. 

    '''
    """Interacts with and learns from the environment."""

    def __init__(self,
                 state_size,
                 action_size,
                 hyperparams,
                 seed=13):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        # Noise process
        self.noises = [OUNoise(action_size, seed) for _ in range (20)]

        self.beta_gen = annealing_generator(start=hyperparams['beta_start'],
                                            end=hyperparams['beta_end'],
                                            factor=hyperparams['beta_factor'])
        # Actor/Critic
        
        # self.actor_target = ActorNetwork(state_size,
        #                                  action_size,
        #                                  hyperparams['actor_topology'],
        #                                  seed).to(device)

        # self.actor_local = ActorNetwork(state_size,
        #                                 action_size,
        #                                 hyperparams['actor_topology'],
        #                                 seed).to(device)

        # self.critic_target = CriticNetwork(state_size=state_size,
        #                                    action_size=action_size,
        #                                    hidden_layer_state_leg = hyperparams['critic_state_leg_topology'],
        #                                    hidden_layer_actions_leg = hyperparams['critic_action_leg_topology'],
        #                                    hidden_layer_head = hyperparams['critic_state_action_leg_topology'],
        #                                    seed=seed).to(device)
        # self.critic_local = CriticNetwork(state_size=state_size,
        #                                    action_size=action_size,
        #                                    hidden_layer_state_leg = hyperparams['critic_state_leg_topology'],
        #                                    hidden_layer_actions_leg = hyperparams['critic_action_leg_topology'],
        #                                    hidden_layer_head = hyperparams['critic_state_action_leg_topology'],
        #                                    seed=seed).to(device)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, seed).to(device)
        self.critic_target = Critic(state_size, action_size, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=0)

        # self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=hyperparams['actor_lr'])
        # self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=hyperparams['critic_lr'])

        # Replay memory
        self.batch_size = hyperparams.get('batch_size', BATCH_SIZE)

        self.memory = PrioritizedReplayBuffer(BUFFER_SIZE,
                                              self.batch_size,
                                              seed,
                                              per_epsilon=hyperparams.get(
                                                  'per_epsilon'),
                                              per_alpha=hyperparams.get('per_alpha'))

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

        self.critic_criterion = nn.MSELoss(reduce=False)

        self.gamma = GAMMA


    def step(self,
             states,
             actions,
             rewards: float,
             next_states: torch.Tensor,
             dones: bool):
        '''
        Function to be called after every interaction between the agent
        and the environment.
        
        Updates the memory and learns.
        '''
        # Save experience in replay memory
        for state,action,reward, next_state, done in zip(states,actions,rewards,next_states,dones): 
            self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % (UPDATE_EVERY*20)

        # If enough samples are available in memory, get random subset and learn
        if len(self.memory) > self.batch_size and self.t_step ==0:
            for _ in range(LEARN_FOR):
                self.learn()

    def act(self,states):
        # import pdb;pdb.set_trace()
        return np.array([self.act_state(state,noise) for state,noise in zip(states,self.noises)]).reshape(20,4)

    def act_state(self,
            state: np.array,
            noise,
            training: bool = True) -> torch.Tensor:
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            training (bool): whether the agent is training or not.
        """
        state = torch.from_numpy(state).float().to(device)

        self.actor_local.eval() 
        with torch.no_grad():
            output_actions = self.actor_local.forward(state.unsqueeze(0)).detach().cpu().numpy()

        if training:
            self.beta = next(self.beta_gen)
            output_actions += noise.sample()
        
        return np.clip(output_actions,-1,1)
        
    def reset(self):
        for noise in self.noises:
            noise.reset()

    def learn(self):

        self.actor_local.train() 

        # 1) Sample experience tuples.

        memory_indices, priorities, experiences = self.memory.sample()
        states, mem_actions, rewards, next_states, dones = experiences

        # 2) Optimize the critic.
        
        self.critic_optimizer.zero_grad()

        # 2.1 Use the actor target network for estimating the actions 
        # and calculate their value using the critic local network.

        critic_output = self.critic_local.forward(states,mem_actions)

        # 2.2 Use the critic target network for using the estimated value.

        with torch.no_grad():
            target_actions = self.actor_target.forward(next_states)

        self.critic_target.eval()
        with torch.no_grad():
            critic_next_action_estimated_values = self.critic_target(next_states,target_actions)

        critic_estimated_values=rewards + (1-dones)*self.gamma*critic_next_action_estimated_values

        if not ~np.isnan((critic_output-critic_estimated_values).detach().numpy()).all():
            import pdb;pdb.set_trace()

        self.memory.update_batches(memory_indices, (critic_output-critic_estimated_values))

        # 2.2) Prioritized replay bias adjustment.

        beta = self.beta

        bias_correction = ((len(self.memory)/len(self.memory))*(1/priorities))**beta
        # print('Bias correction',priorities,beta,bias_correction,bias_correction/torch.max(bias_correction))
        
        bias_correction = bias_correction/torch.max(bias_correction)
        loss = (self.critic_criterion(critic_output, critic_estimated_values)*bias_correction).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # 3) Optimize the actor.

        self.actor_optimizer.zero_grad()

        local_actions = self.actor_local.forward(states)

        loss = (-self.critic_local.forward(states,local_actions)*bias_correction).mean()
        loss.backward()
        self.actor_optimizer.step()

        # ------------------- update target network ------------------- #
        if self.t_step == 0:
            self.soft_update(self.actor_local, self.actor_target, TAU)
            self.soft_update(self.critic_local, self.critic_target, TAU)

    def soft_update(self,
                    local_model: nn.Module,
                    target_model: nn.Module,
                    tau: float):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        This is an alterative to the original formulation of the DQN 
        paper, in which the target agent is updated with the local 
        model every X steps.
        
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., starting_theta=0.3, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = starting_theta
        self.theta_gen = annealing_generator(starting_theta,0.05,0.9995)
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)
        self.theta = next(self.theta_gen)
        # print('theta:',self.theta)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


class SumTree(object):
    '''
    SumTree for efficiently performing weighted sampling. 

    Adapted from https://pylessons.com/CartPole-PER/
    '''

    # Here we initialize the tree with all nodes = 0, and initialize the data with all values = 0
    def __init__(self, capacity):
        # Number of leaf nodes (final nodes) that contains experiences
        self.capacity = capacity

        # Generate the tree with all nodes values = 0
        # To understand this calculation (2 * capacity - 1) look at the schema below
        # Remember we are in a binary node (each node has max 2 children) so 2x size of leaf (capacity) - 1 (root node)
        # Parent nodes = capacity - 1
        # Leaf nodes = capacity
        self.tree = np.zeros(2 * capacity - 1)

        self.data_pointer = 0  # Pointer to the next leave to update.

        # Contains the experiences (so the size of data is capacity)
        self.data = [None]*capacity

    def add(self, priority, data):
        # Look at what index we want to put the experience
        tree_index = self.data_pointer + (self.capacity - 1)

        """ tree:
                    0
                   / \
                  0   0
                 / \ / \
        tree_index  0 0  0  We fill the leaves from left to right
        """

        # Update data frame
        self.data[self.data_pointer] = data

        # Update the leaf
        self.update(tree_index, priority)

        # Add 1 to data_pointer
        self.data_pointer += 1

        # If we're above the capacity, we go back to first index (we overwrite)
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_index, priority):
        # Change = new priority score - former priority score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        # then propagate the change through tree
        # this method is faster than the recursive loop
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, v):
        parent_index = 0

        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0]  # Returns the root node

    @property
    def maximum_priority(self):
        return np.max(self.tree[-self.capacity:])  # Returns the root node

    def __len__(self):
        """Return the current size of internal memory."""
        return np.sum(~(self.tree[-self.capacity:] == 0))


class PrioritizedReplayBuffer:
    """Fixed-size buffer to store experience tuples.
       Leverages a SumTree for efficiently sampling."""

    def __init__(self,
                 buffer_size,
                 batch_size,
                 seed,
                 per_epsilon: float = None,
                 per_alpha: float = None,):
        """Initialize a PrioritizedReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.tree = SumTree(buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=[
                                     "state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.per_epsilon = per_epsilon or 0.001
        self.per_alpha = per_alpha or 0

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        maximum_priority = self.tree.maximum_priority + \
            self.per_epsilon  # TODO use clipped abs error?
        if maximum_priority == 0:
            maximum_priority = 1
        # print(maximum_priority)
        self.tree.add(maximum_priority, e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""

        experiences = []
        indices = []
        priorities = []
        # We divide the priority into buckets and sample from each of those
        segments = self.tree.total_priority/self.batch_size
        values = []
        for i in range(self.batch_size):
            value = random.uniform(i*segments, (i+1)*segments)
            leaf_index, priority, data = self.tree.get_leaf(value)

            experiences.append(data)
            indices.append(leaf_index)
            priorities.append(priority)
            values.append(value)

        try:
            states = torch.from_numpy(
                np.vstack([e.state for e in experiences if e is not None])).float().to(device)
            actions = torch.from_numpy(
                np.vstack([e.action for e in experiences if e is not None])).float().to(device)
            rewards = torch.from_numpy(
                np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
            next_states = torch.from_numpy(np.vstack(
                [e.next_state for e in experiences if e is not None])).float().to(device)
            dones = torch.from_numpy(np.vstack(
                [e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        except Exception as e:
            import pdb;pdb.set_trace()

        return indices, torch.Tensor(priorities).to(device), (states, actions, rewards, next_states, dones)

    def update_batches(self, indices, errors):

        for index, error in zip(indices, errors.detach().cpu().numpy()):
            self.tree.update(
                index, (abs(error)+self.per_epsilon)**self.per_alpha)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.tree)
