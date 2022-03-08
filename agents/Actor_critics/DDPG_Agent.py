
import gym
from os.path import join as joindir
import torch
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple
from utilities.Replay_Buffer import ReplayBuffer
from agents.Actor_critics.Base_Agent import BaseAgent
from utilities.Actor_Crtics_utilities.DDPG.OU_Noise import *
from utilities.Actor_Crtics_utilities.DDPG.DDPG_Network import *
import numpy as np
import matplotlib.pyplot as plt
import datetime
import matplotlib
matplotlib.use('agg')
import seaborn as sns
import pandas as pd
import numpy as np




datestr = datetime.datetime.now().strftime('%Y-%m-%d')


class DDPGAgent(BaseAgent):
    agent_name = "DDPG"
    
    """DDPG Agent interacting with environment which is a BaseAgent super class.
    
    Atribute:
        Parameters from Base Agent:
    
        env (gym.Env): openAI Gym environment
        
        actor (nn.Module): target actor model to select actions
        actor_target (nn.Module): actor model to predict next actions
        actor_optimizer (Optimizer): optimizer for training actor
        critic (nn.Module): critic model to predict state values
        critic_target (nn.Module): target critic model to predict state values
        critic_optimizer (Optimizer): optimizer for training critic
        
        memory (ReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        tau (float): parameter for soft target update
        gamma (float): discount factor
        device (torch.device): cpu / gpu
        transition (list): temporory storage for the recent transition
        total_step (int): total step numbers
        initial_random_steps (int): initial random action steps
        is_test (bool): flag to show the current mode (train / test)
        
        DDPG Sub-class of Basic Agent:
        noise (OUNoise): noise generator for exploration ou_noise_theta, ou_noise_sigma
        
    """
        
    def __init__(
        
        self,
        env: gym.Env,
        memory_size: int,
        batch_size: int,
        ou_noise_theta: float,
        ou_noise_sigma: float,
        epsilon: float,
        epoch: int,
        rollout_len: int,
        entropy_weight: float,
        gamma: float = 0.99,
        tau: float = 5e-3,
        initial_random_steps: int = 1e4,
        policy_update_freq: int = 2,
        exploration_noise: float = 0.1,
        target_policy_noise: float = 0.2,
        target_policy_noise_clip: float = 0.5,
        
    ):    
        
        super().__init__(            
        env,
        memory_size,
        batch_size,
        gamma,
        tau,
        initial_random_steps,
        entropy_weight,
        #DDPG parameters sub-class
        ou_noise_theta,
        ou_noise_sigma)
       
       
       
         # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print('Device used: {}' .format(self.device))
    
        
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        
        
        self.env = env
        self.memory = ReplayBuffer(obs_dim, memory_size, batch_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.initial_random_steps = initial_random_steps
     
        
        
        
        # transition (state, log_prob, next_state, reward, done)
        self.transition: list = list()
        
        # total steps count
        self.total_step = 0

        # mode: train / test
        self.is_test = False
        
         # noise
        self.noise = OUNoise(
            action_dim,
            theta=ou_noise_theta,
            sigma=ou_noise_sigma,
        )
        
         # networks
        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.actor_target = Actor(obs_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(obs_dim + action_dim).to(self.device)
        self.critic_target = Critic(obs_dim + action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        
    
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, done, _ = self.env.step(action)
        
        if not self.is_test:
            self.transition += [reward, next_state, done]
            self.memory.store(*self.transition)
    
        return next_state, reward, done
    
    
    
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # if initial random action should be conducted
        if self.total_step < self.initial_random_steps and not self.is_test:
            selected_action = self.env.action_space.sample()
        else:
            selected_action = self.actor(
                torch.FloatTensor(state).to(self.device)
            ).detach().cpu().numpy()
        
        # add noise for exploration during training
        if not self.is_test:
            noise = self.noise.sample()
            selected_action = np.clip(selected_action + noise, -1.0, 1.0)
        
        self.transition = [state, selected_action]
        
        return selected_action
    
    
    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        device = self.device  # for shortening the following lines
        
        samples = self.memory.sample_batch()
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.FloatTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
        
        masks = 1 - done
        next_action = self.actor_target(next_state)
        next_value = self.critic_target(next_state, next_action)
        curr_return = reward + self.gamma * next_value * masks
        
        # train critic
        values = self.critic(state, action)
        critic_loss = F.mse_loss(values, curr_return)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
                
        # train actor
        actor_loss = -self.critic(state, self.actor(state)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # target update
        self._target_soft_update()
        
        return actor_loss.data, critic_loss.data
    
    
    def _target_soft_update(self):
        """Soft-update: target = tau*local + (1-tau)*target."""
        tau = self.tau
        
        for t_param, l_param in zip(
            self.actor_target.parameters(), self.actor.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)
            
        for t_param, l_param in zip(
            self.critic_target.parameters(), self.critic.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)
    
        
    def train_rst(self, num_frames: int, plotting_interval: int = 200):
        """Train the agent."""
        self.is_test = False
        
        state = self.env.reset()
        
        actor_losses = []
        critic_losses = []
        scores = []
        score = 0
        reward_record = []

       
        for self.total_step in range(1, num_frames + 1):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward
            
            # if episode ends
            if done:         
                state = self.env.reset()
                scores.append(score)
                score = 0
            
            # if training is ready
            if (
                len(self.memory) >= self.batch_size 
                #and self.total_step > self.initial_random_steps
            ):
                actor_loss, critic_loss = self.update_model()
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)

                actor_loss = actor_loss.cpu().detach().numpy()
                critic_loss = critic_loss.cpu().detach().numpy()
                
                reward_record.append({
                'Agent' : self.agent_name,
                'steps': self.total_step, 
                'Reward': score, 
                'meanepreward': np.mean(scores),
                'stdepreward': np.std(scores)*0.1,
                'Actor loss' : actor_loss,
                'Critic loss' : critic_loss})
                  
        
            if self.total_step % plotting_interval == 0:
                print('Finished episode: {} Average Reward: {:.4f} STD Reward: {:.4f} Actor Loss : {:.4f} Critic Loss : {:.4f} ' \
                .format(self.total_step, reward_record[-1]['meanepreward'], reward_record[-1]['stdepreward'],actor_loss, critic_loss))
                print('-----------------')
        
        
        self._plot_record(reward_record)
        
                
        self.env.close()
        
        
        return reward_record
            
            
            
    def _plot_record(
        self, 
        reward_record: List[float], 
            
    ):

        reward_record = pd.DataFrame(reward_record)
        reward_record['reward_smooth'] = reward_record['meanepreward'].ewm(span=200).mean()
        """Plot the training progresses."""        
        
        
        sns.set_theme()
        sns.set_style("darkgrid")
        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.plot(reward_record['steps'], reward_record['meanepreward'], label='trajectory reward')
        plt.plot(reward_record['steps'], reward_record['reward_smooth'], label='smoothed reward')
        plt.fill_between(reward_record['steps'], reward_record['meanepreward'] - reward_record['stdepreward'], 
            reward_record['meanepreward'] + reward_record['stdepreward'], color='b', alpha=0.2)
        plt.legend()
        plt.xlabel('steps of env interaction (sample complexity)')
        plt.ylabel('average reward')
        plt.title('{} on {}'.format(self.agent_name,self.env.unwrapped.spec.id))
        
        
        plt.subplot(132)
        plt.plot(reward_record['steps'], reward_record['Actor loss'], label='loss')
        plt.legend()
        plt.xlabel('steps')
        plt.ylabel('loss')
        plt.title('Actor Loss')

        plt.subplot(133)
        plt.plot(reward_record['steps'], reward_record['Critic loss'], label='loss')
        plt.legend()
        plt.xlabel('steps')
        plt.ylabel('loss')
        plt.title('Critic Loss')
        
        
        
        
        #plt.show()
        
        reward_record.to_csv(joindir(RESULT_DIR, '{}-record-{}-{}.csv'.format(self.agent_name,self.env.unwrapped.spec.id, datestr)))
        plt.savefig(joindir(RESULT_DIR, '{}-{}-{}.pdf'.format(self.agent_name,self.env.unwrapped.spec.id, datestr)))
                                
    
    