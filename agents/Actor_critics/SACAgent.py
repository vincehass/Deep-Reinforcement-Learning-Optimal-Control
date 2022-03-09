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
from utilities.Actor_Crtics_utilities.SAC.SAC_Network import *
import matplotlib.pyplot as plt
import datetime
import matplotlib
matplotlib.use('agg')
import seaborn as sns
import pandas as pd
import numpy as np
from utilities.Config import RESULT_DIR



datestr = datetime.datetime.now().strftime('%Y-%m-%d')



class SACAgent(BaseAgent):
    agent_name = "SAC"
    
    """SAC Agent interacting with environment which is a BaseAgent super class.
    
    Atribute:
    Shared parameters from Base Agent:
    
        env (gym.Env): openAI Gym environment
        
        actor (nn.Module): actor model to select actions
        actor_optimizer (Optimizer): optimizer for training actor
        vf (nn.Module): critic model to predict state values
        vf_target (nn.Module): target critic model to predict state values
        vf_optimizer (Optimizer): optimizer for training vf
        
        qf_1 (nn.Module): critic model to predict state-action values
        qf_2 (nn.Module): critic model to predict state-action values
        qf_1_optimizer (Optimizer): optimizer for training qf_1
        qf_2_optimizer (Optimizer): optimizer for training qf_2
        
        memory (ReplayBuffer): replay memory
        batch_size (int): batch size for sampling
        gamma (float): discount factor
        tau (float): parameter for soft target update
        initial_random_steps (int): initial random action steps
        policy_update_freq (int): policy update frequency
        device (torch.device): cpu / gpu
        
        SAC Sub-class of Basic Agent:
        target_entropy (int): desired entropy used for the inequality constraint
        log_alpha (torch.Tensor): weight for entropy
        alpha_optimizer (Optimizer): optimizer for alpha
        transition (list): temporory storage for the recent transition
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
        
          
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
        policy_update_freq,
        rollout_len,
        entropy_weight)
        
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
         # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print('Device used: {}' .format(self.device))
        
         # transition (state, log_prob, next_state, reward, done)
        self.transition: list = list()
        
        # total steps count
        self.total_step = 0

        # mode: train / test
        self.is_test = False
        
        
        
        
        
        #Initailization
        
        self.env = env
        self.memory = ReplayBuffer(obs_dim, memory_size, batch_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.policy_update_freq = policy_update_freq
        self.initial_random_steps = initial_random_steps
        
       # automatic entropy tuning
        self.target_entropy = -np.prod((action_dim,)).item()  # heuristic
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)

        # actor
        self.actor = Actor(obs_dim, action_dim).to(self.device)
        
        # v function
        self.vf = CriticV(obs_dim).to(self.device)
        self.vf_target = CriticV(obs_dim).to(self.device)
        self.vf_target.load_state_dict(self.vf.state_dict())
        
        # q function
        self.qf_1 = CriticQ(obs_dim + action_dim).to(self.device)
        self.qf_2 = CriticQ(obs_dim + action_dim).to(self.device)

        # optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.vf_optimizer = optim.Adam(self.vf.parameters(), lr=3e-4)
        self.qf_1_optimizer = optim.Adam(self.qf_1.parameters(), lr=3e-4)
        self.qf_2_optimizer = optim.Adam(self.qf_2.parameters(), lr=3e-4)
        
        # transition to store in memory
        self.transition = list()
        
        # total steps count
        self.total_step = 0

        # mode: train / test
        self.is_test = False
    
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
            )[0].detach().cpu().numpy()
            
        self.transition = [state, selected_action]
        
        return selected_action
    
            
        
    def update_model(self) -> Tuple[torch.Tensor, ...]:
        """Update the model by gradient descent."""
        device = self.device  # for shortening the following lines
        
        samples = self.memory.sample_batch()
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.FloatTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
        new_action, log_prob = self.actor(state)
        
        # train alpha (dual problem)
        alpha_loss = (
            -self.log_alpha.exp() * (log_prob + self.target_entropy).detach()
        ).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        alpha = self.log_alpha.exp()  # used for the actor loss calculation
        
        # q function loss
        mask = 1 - done
        q_1_pred = self.qf_1(state, action)
        q_2_pred = self.qf_2(state, action)
        v_target = self.vf_target(next_state)
        q_target = reward + self.gamma * v_target * mask
        qf_1_loss = F.mse_loss(q_1_pred, q_target.detach())
        qf_2_loss = F.mse_loss(q_2_pred, q_target.detach())
        
        # v function loss
        v_pred = self.vf(state)
        q_pred = torch.min(
            self.qf_1(state, new_action), self.qf_2(state, new_action)
        )
        v_target = q_pred - alpha * log_prob
        vf_loss = F.mse_loss(v_pred, v_target.detach())
        
        if self.total_step % self.policy_update_freq == 0:
            # actor loss
            advantage = q_pred - v_pred.detach()
            actor_loss = (alpha * log_prob - advantage).mean()
            
            # train actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
        
            # target update (vf)
            self._target_soft_update()
        else:
            actor_loss = torch.zeros(1)
            
        # train Q functions
        self.qf_1_optimizer.zero_grad()
        qf_1_loss.backward()
        self.qf_1_optimizer.step()

        self.qf_2_optimizer.zero_grad()
        qf_2_loss.backward()
        self.qf_2_optimizer.step()
        
        qf_loss = qf_1_loss + qf_2_loss

        # train V function
        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()
        
        return actor_loss.data, qf_loss.data, vf_loss.data, alpha_loss.data
    
        
    def _target_soft_update(self):
        """Soft-update: target = tau*local + (1-tau)*target."""
        tau = self.tau
        
        for t_param, l_param in zip(
            self.vf_target.parameters(), self.vf.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)
        
    def train_rst(self, num_frames: int, plotting_interval: int = 200):
        """Train the agent."""
        self.is_test = False
        
        state = self.env.reset()
        
        actor_losses, qf_losses, vf_losses, alpha_losses = [], [], [], []
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
                losses = self.update_model()
                actor_losses.append(losses[0])
                qf_losses.append(losses[1])
                vf_losses.append(losses[2])
                alpha_losses.append(losses[3])
                
                reward_record.append({
                'Agent' : self.agent_name,
                'steps': self.total_step, 
                'Reward': score, 
                'meanepreward': np.mean(scores),
                'stdepreward': np.std(scores)*0.1,
                'Qf loss' : losses[1],
                'Vf loss' : losses[2],
                'alpha loss' : losses[3],})
                  
                #print(losses[1], losses[2], losses[3])
            if self.total_step % plotting_interval == 0:
                print('Finished episode: {} Average Reward: {:.4f} STD Reward: {:.4f} Qf Loss : {:.4f} Vf Loss : {:.4f} alpha Loss : {:.4f}' \
                .format(self.total_step, reward_record[-1]['meanepreward'], reward_record[-1]['stdepreward'],losses[1], losses[2], losses[3]))
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
        plt.plot(reward_record['steps'], reward_record['Qf loss'], label='Qf Loss')
        plt.legend()
        plt.xlabel('steps')
        plt.ylabel('loss')
        plt.title('Q-function Loss')

        plt.subplot(132)
        plt.plot(reward_record['steps'], reward_record['Vf loss'], label='Vf loss')
        plt.legend()
        plt.xlabel('steps')
        plt.ylabel('loss')
        plt.title('Value function Loss')
        
        
        plt.subplot(133)
        plt.plot(reward_record['steps'], reward_record['alpha loss'], label='alpha loss')
        plt.legend()
        plt.xlabel('steps')
        plt.ylabel('loss')
        plt.title('Alpha Loss')
        
        
        
        #plt.show()
        
        reward_record.to_csv(joindir(RESULT_DIR, '{}-record-{}-{}.csv'.format(self.agent_name,self.env.unwrapped.spec.id, datestr)))
        plt.savefig(joindir(RESULT_DIR, '{}-{}-{}.png'.format(self.agent_name,self.env.unwrapped.spec.id, datestr)))
                                
    
    