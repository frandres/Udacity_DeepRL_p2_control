import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class ActorNetwork(nn.Module):
    """Actor (Policy) Model.
       Skeleton adapted from Udacity exercise sample code. 
       Network that maps state -> action, assuming action 
       is a continuous value in the [1,1] range.
    """

    def __init__(self,
                 state_size,
                 action_size,
                 hidden_layers,
                 seed,
                 ):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)

        self.hidden_layers = nn.ModuleList(
            [nn.Linear(state_size, hidden_layers[0])])
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2)
                                  for h1, h2 in layer_sizes])

        self.batch_normalization = nn.ModuleList([nn.BatchNorm1d(layer.in_features) for layer in self.hidden_layers])

        for layer in self.hidden_layers:
            layer.weight.data.uniform_(*hidden_init(layer))

        self.output = nn.Linear(hidden_layers[-1], action_size)
        self.output_batch_normalization = nn.BatchNorm1d(hidden_layers[-1])
        self.output.weight.data.uniform_(*hidden_init(layer))


    def forward(self, state):
        
        x = state
        for linear,batch_norm in zip(self.hidden_layers,self.batch_normalization):
            x = F.relu(linear(batch_norm(x)))

        x = self.output(self.output_batch_normalization(x))

        return  torch.tanh(x)

class CriticNetwork(nn.Module):
    """Critic (Value) Model.
       Skeleton adapted from Udacity exercise sample code. 
       Network that maps state,action -> value
    """

    def __init__(self,
                 state_size:int,
                 action_size:int,
                 hidden_layer_state_leg:List[int],
                 hidden_layer_actions_leg:List[int],
                 hidden_layer_head:List[int],
                 seed,
                 ):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden_layer_state_leg (List[int]): 
                A list of hidden layers with the number of neurons in each layer, 
                for processing the state input.
            hidden_layer_actions_leg (List[int]): 
                A list of hidden layers with the number of neurons in each layer, 
                for processing the action input.
            hidden_layer_head (List[int]): 
                A list of hidden layers with the number of neurons in each layer, 
                for processing the concatenation of the state and actions.
            seed (int): Random seed
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)

    
        self.state_hidden_layers,self.state_batch_normalization = self.build_leg(state_size, hidden_layer_state_leg)
        self.action_hidden_layers,self.action_batch_normalization = self.build_leg(action_size, hidden_layer_actions_leg)

        action_size_output = hidden_layer_actions_leg[-1] if len(hidden_layer_actions_leg)>0 else action_size
        state_size_output = hidden_layer_state_leg[-1] if len(hidden_layer_state_leg)>0 else state_size
        
        self.head_layers,self.head_batch_normalization = self.build_leg(action_size_output+state_size_output, hidden_layer_head)
        # Add a variable number of more hidden layers
        assert len(hidden_layer_head)>0
        self.output = nn.Linear(hidden_layer_head[-1], 1)

    def build_leg(self,input_size:int,hidden_layers:List[int]):
        if len(hidden_layers) ==0:
            return [],[]

        hidden_layers_list = nn.ModuleList(
            [nn.Linear(input_size, hidden_layers[0])])

        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        hidden_layers_list.extend([nn.Linear(h1, h2)
                              for h1, h2 in layer_sizes])

        for layer in hidden_layers_list:
            layer.weight.data.uniform_(*hidden_init(layer))

        batch_normalization = nn.ModuleList([nn.BatchNorm1d(layer.in_features) for layer in hidden_layers_list])

        return hidden_layers_list,batch_normalization

    def forward(self, state,action):
        """Build a network that maps state -> action values."""
        for linear,batch_norm in zip(self.state_hidden_layers,self.state_batch_normalization):
            state = F.relu(linear(batch_norm(state)))

        for linear,batch_norm in zip(self.action_hidden_layers,self.action_batch_normalization):
            action = F.relu(linear(batch_norm(action)))

        state_action = torch.cat((state,action),dim=-1)

        for linear,batch_norm in zip(self.head_layers,self.head_batch_normalization):
            state_action = F.relu(linear(batch_norm(state_action)))
        return self.output(state_action)

        