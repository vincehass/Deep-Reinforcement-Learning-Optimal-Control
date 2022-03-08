import torch
from torch import optim 
import gym
import numpy as np
from typing import Tuple
from utilities.Replay_Buffer import ReplayBuffer
from utilities.DQN_utilities.Prioritized_Replay_Buffer import PrioritizedReplayBuffer
from utilities.DQN_utilities.Replay_Buffer_NStep import ReplayBufferNStep
from utilities.DQN_utilities.DQN_Network import Network




class BaseAgent(object):
    """Basic Agent interacting with environment.
    
    Attribute:
        env (gym.Env): openAI Gym environment
        memory (ReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        epsilon (float): parameter for epsilon greedy policy
        epsilon_decay (float): step size to decrease epsilon
        max_epsilon (float): max value of epsilon
        min_epsilon (float): min value of epsilon
        target_update (int): period for target model's hard update
        gamma (float): discount factor
        dqn (Network): model to train and select actions
        dqn_target (Network): target model to update
        optimizer (torch.optim): optimizer for training dqn
        transition (list): transition information including 
                           state, action, reward, next_state, done
    """

    def __init__(
        self, 
        env: gym.Env,
        memory_size: int,
        batch_size: int,
        target_update: int,
        epsilon_decay: float,
        prioritized_memory: bool,
        n_step: int,
        use_n_step: bool = False,
        max_epsilon: float = 1.0,
        min_epsilon: float = 0.1,
        gamma: float = 0.99,
        
       
        
    ):
        """Initialization.
        
        Args:
            env (gym.Env): openAI Gym environment
            memory_size (int): length of memory
            batch_size (int): batch size for sampling
            target_update (int): period for target model's hard update
            epsilon_decay (float): step size to decrease epsilon
            lr (float): learning rate
            max_epsilon (float): max value of epsilon
            min_epsilon (float): min value of epsilon
            gamma (float): discount factor
        """
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        self.env = env
        self.prioritized_memory = prioritized_memory
        self.n_step = n_step
        self.use_n_step = use_n_step
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.target_update = target_update
        self.gamma = gamma
        
        
        
        self.use_n_step = True if n_step > 1 else False
        # print(self.n_step, self.use_n_step, n_step)
        # print(self.prioritized_memory)
        if prioritized_memory:
            self.memory = PrioritizedReplayBuffer(obs_dim, memory_size, batch_size)
        
        else:
            
            #self.memory = ReplayBuffer(obs_dim, memory_size, batch_size)
            
            
            if self.use_n_step:
                self.n_step = n_step
                self.memory = ReplayBufferNStep(
                obs_dim, memory_size, batch_size, n_step=n_step, gamma=gamma
                )
            else:
                self.memory = ReplayBuffer(obs_dim, memory_size, batch_size)
        
        
        
        
        
        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print('Device used: {}' .format(self.device))

        # networks: dqn, dqn_target
        self.dqn = Network(obs_dim, action_dim).to(self.device)
        self.dqn_target = Network(obs_dim, action_dim).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()
        
        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters())

        # transition to store in memory
        self.transition = list()
        
        # mode: train / test
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # epsilon greedy policy
        if self.epsilon > np.random.random():
            selected_action = self.env.action_space.sample()
        else:
            selected_action = self.dqn(
                torch.FloatTensor(state).to(self.device)
            ).argmax()
            selected_action = selected_action.detach().cpu().numpy()
        
        if not self.is_test:
            self.transition = [state, selected_action]
        
        return selected_action
    
    
    
    

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, done, _ = self.env.step(action)

        if not self.is_test:
            self.transition += [reward, next_state, done]
            #self.memory.store(*self.transition)
            # N-step transition
            if self.use_n_step:
                one_step_transition = self.memory.store(*self.transition)
            # 1-step transition
            else:
                one_step_transition = self.transition

            # add a single step transition
            if one_step_transition:
                self.memory.store(*one_step_transition)
    
        return next_state, reward, done
    

            
